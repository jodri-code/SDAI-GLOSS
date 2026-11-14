@file:Suppress("UNCHECKED_CAST")

package com.shifthackz.aisdv1.feature.onnx.ai.vae

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtException
import ai.onnxruntime.OrtSession.SessionOptions
import ai.onnxruntime.providers.NNAPIFlags
import android.graphics.Bitmap
import com.shifthackz.aisdv1.feature.onnx.LocalDiffusionContract.KEY_SAMPLE
import com.shifthackz.aisdv1.feature.onnx.LocalDiffusionContract.ORT
import com.shifthackz.aisdv1.feature.onnx.LocalDiffusionContract.ORT_KEY_MODEL_FORMAT
import com.shifthackz.aisdv1.feature.onnx.LocalDiffusionContract.VAE_SCALING_FACTOR
import com.shifthackz.aisdv1.feature.onnx.entity.Array3D
import com.shifthackz.aisdv1.feature.onnx.entity.LocalDiffusionFlag
import com.shifthackz.aisdv1.feature.onnx.environment.DeviceNNAPIFlagProvider
import com.shifthackz.aisdv1.feature.onnx.environment.OrtEnvironmentProvider
import java.io.File
import java.nio.FloatBuffer
import java.util.EnumSet

/**
 * Turns an input image into Stable Diffusion latents for Img2Img.
 *
 * Mirrors [VaeDecoder]: the ONNX session lives only for the duration of [encode], so it never
 * coexists in memory with the UNet session.
 *
 * Expected model contract (matches the Optimum-style `vae_decoder`):
 * - input `sample`: float32 `[batch, 3, height, width]`, pixel values in `[-1, 1]`
 * - output 0: float32 `[batch, 4, height / 8, width / 8]` raw (unscaled) latents.
 *   Outputs with more than 4 channels (`latent_parameters`: mean + logvar) are accepted,
 *   and only the first 4 channels (the mean) are used.
 */
internal class VaeEncoder(
    private val ortEnvironmentProvider: OrtEnvironmentProvider,
    private val deviceNNAPIFlagProvider: DeviceNNAPIFlagProvider,
    private val modelLocator: VaeEncoderModelLocator,
) {

    /**
     * Encodes [bitmap] into UNet-ready latents (already multiplied by [VAE_SCALING_FACTOR]).
     *
     * The bitmap is stretched to [width] x [height]; both must be multiples of 8.
     *
     * @return latents shaped `[1][4][height / 8][width / 8]`.
     */
    fun encode(bitmap: Bitmap, width: Int, height: Int): Array3D<FloatArray> {
        require(width > 0 && height > 0 && width % 8 == 0 && height % 8 == 0) {
            "Img2Img size must be a positive multiple of 8, got ${width}x${height}."
        }
        val environment = ortEnvironmentProvider.get()
        val modelPath = modelLocator.resolve()
        val options = SessionOptions()
        options.addConfigEntry(ORT_KEY_MODEL_FORMAT, ORT)
        if (deviceNNAPIFlagProvider.get() == LocalDiffusionFlag.NN_API.value) {
            options.addNnapi(EnumSet.of(NNAPIFlags.CPU_DISABLED))
        }
        val latents = options.use {
            val session = try {
                environment.createSession(modelPath, options)
            } catch (e: OrtException) {
                throw IllegalStateException(
                    "Unable to load the VAE encoder at $modelPath (${File(modelPath).length()} bytes): ${e.message}",
                    e,
                )
            }
            session.use {
                bitmapToTensor(bitmap, width, height).use { input ->
                    session.run(mapOf(KEY_SAMPLE to input)).use { result ->
                        toScaledLatents(result[0].value as Array3D<FloatArray>)
                    }
                }
            }
        }
        return latents
    }

    private fun bitmapToTensor(bitmap: Bitmap, width: Int, height: Int): OnnxTensor {
        val scaled = if (bitmap.width == width && bitmap.height == height) {
            bitmap
        } else {
            Bitmap.createScaledBitmap(bitmap, width, height, true)
        }
        val pixels = IntArray(width * height)
        scaled.getPixels(pixels, 0, width, 0, 0, width, height)
        if (scaled !== bitmap) scaled.recycle()

        // NCHW, normalized to [-1, 1]
        val plane = width * height
        val data = FloatArray(3 * plane)
        for (i in 0 until plane) {
            val pixel = pixels[i]
            data[i] = ((pixel shr 16) and 0xFF) / 127.5f - 1f
            data[plane + i] = ((pixel shr 8) and 0xFF) / 127.5f - 1f
            data[2 * plane + i] = (pixel and 0xFF) / 127.5f - 1f
        }
        return OnnxTensor.createTensor(
            ortEnvironmentProvider.get(),
            FloatBuffer.wrap(data),
            longArrayOf(1, 3, height.toLong(), width.toLong()),
        )
    }

    private fun toScaledLatents(raw: Array3D<FloatArray>): Array3D<FloatArray> {
        require(raw.isNotEmpty() && raw[0].size >= LATENT_CHANNELS) {
            "Unexpected VAE encoder output: expected at least $LATENT_CHANNELS channels."
        }
        return Array(raw.size) { b ->
            Array(LATENT_CHANNELS) { c ->
                Array(raw[b][c].size) { y ->
                    FloatArray(raw[b][c][y].size) { x -> raw[b][c][y][x] * VAE_SCALING_FACTOR }
                }
            }
        }
    }

    private companion object {
        const val LATENT_CHANNELS = 4
    }
}
