package com.shifthackz.aisdv1.feature.onnx.ai.vae

import com.shifthackz.aisdv1.domain.entity.DownloadState
import com.shifthackz.aisdv1.domain.entity.NetworkUsageBucket
import com.shifthackz.aisdv1.domain.repository.NetworkUsageRepository
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.io.IOException
import java.security.MessageDigest
import java.util.concurrent.TimeUnit

/**
 * Downloads the shared Img2Img VAE encoder once.
 *
 * Tries [VaeEncoderDistribution.SOURCES] in order. A source only counts if the file's SHA-256
 * matches [VaeEncoderDistribution.SHA256]; otherwise the next source is tried. The file is
 * written to a `.part` file and moved into place only after it verified, so a half-finished or
 * corrupt download can never be picked up by the encoder.
 */
internal class VaeEncoderDownloader(
    private val modelLocator: VaeEncoderModelLocator,
    private val networkUsageRepository: NetworkUsageRepository,
) {

    private val httpClient: OkHttpClient = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .writeTimeout(30, TimeUnit.SECONDS)
        .callTimeout(0, TimeUnit.MILLISECONDS)
        .build()

    fun download(): Flow<DownloadState> = flow {
        val sources = VaeEncoderDistribution.SOURCES
        val expectedHash = VaeEncoderDistribution.SHA256.trim().lowercase()
        if (sources.isEmpty() || expectedHash.isEmpty()) {
            emit(DownloadState.Error(IllegalStateException("The image encoder download is not configured in this build.")))
            return@flow
        }

        val target = modelLocator.sharedFile
        target.parentFile?.mkdirs()
        val partial = File(target.parentFile, "${target.name}.part")

        emit(DownloadState.Downloading(0))
        var lastError: Throwable? = null
        for (url in sources) {
            try {
                val hash = fetch(url, partial) { percent -> emit(DownloadState.Downloading(percent)) }
                if (hash != expectedHash) throw IOException("Checksum mismatch for $url")
                target.delete()
                if (!partial.renameTo(target)) throw IOException("Unable to store the encoder at ${target.path}")
                emit(DownloadState.Complete(target.path))
                return@flow
            } catch (e: CancellationException) {
                partial.delete()
                throw e
            } catch (e: Exception) {
                lastError = e
                partial.delete()
                emit(DownloadState.Downloading(0))
            }
        }
        emit(
            DownloadState.Error(
                IllegalStateException(
                    "Unable to download the image encoder from any source (${lastError?.message}).",
                    lastError,
                ),
            ),
        )
    }.flowOn(Dispatchers.IO)

    /** Streams [url] into [destination] and returns the lowercase hex SHA-256 of what was written. */
    private suspend fun fetch(
        url: String,
        destination: File,
        onProgress: suspend (Int) -> Unit,
    ): String {
        val digest = MessageDigest.getInstance("SHA-256")
        var downloadedBytes = 0L
        try {
            httpClient.newCall(Request.Builder().url(url).get().build()).execute().use { response ->
                if (!response.isSuccessful) throw IOException("HTTP ${response.code} from $url")
                val body = response.body
                val totalBytes = body.contentLength()
                val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
                var lastPercent = 0
                body.byteStream().use { input ->
                    destination.outputStream().buffered().use { output ->
                        while (true) {
                            currentCoroutineContext().ensureActive()
                            val read = input.read(buffer)
                            if (read == -1) break
                            output.write(buffer, 0, read)
                            digest.update(buffer, 0, read)
                            downloadedBytes += read
                            if (totalBytes > 0L) {
                                val percent = ((downloadedBytes * 100L) / totalBytes).toInt().coerceIn(0, 100)
                                if (percent != lastPercent) {
                                    lastPercent = percent
                                    onProgress(percent)
                                }
                            }
                        }
                    }
                }
            }
        } finally {
            networkUsageRepository.increment(NetworkUsageBucket.MODEL_DOWNLOADS, downloadedBytes)
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }
}
