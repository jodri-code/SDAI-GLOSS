package com.shifthackz.aisdv1.feature.onnx

import ai.onnxruntime.OnnxTensor
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Base64
import com.shifthackz.aisdv1.core.common.log.debugLog
import com.shifthackz.aisdv1.core.common.log.errorLog
import com.shifthackz.aisdv1.domain.entity.DownloadState
import com.shifthackz.aisdv1.domain.entity.ImageToImagePayload
import com.shifthackz.aisdv1.domain.entity.LocalDiffusionStatus
import com.shifthackz.aisdv1.domain.entity.TextToImagePayload
import com.shifthackz.aisdv1.domain.feature.diffusion.LocalDiffusion
import com.shifthackz.aisdv1.feature.onnx.LocalDiffusionContract.TAG
import com.shifthackz.aisdv1.feature.onnx.ai.tokenizer.LocalDiffusionTextTokenizer
import com.shifthackz.aisdv1.feature.onnx.ai.unet.UNet
import com.shifthackz.aisdv1.feature.onnx.ai.vae.VaeEncoder
import com.shifthackz.aisdv1.feature.onnx.ai.vae.VaeEncoderDistribution
import com.shifthackz.aisdv1.feature.onnx.ai.vae.VaeEncoderDownloader
import com.shifthackz.aisdv1.feature.onnx.ai.vae.VaeEncoderModelLocator
import com.shifthackz.aisdv1.feature.onnx.environment.OrtEnvironmentProvider
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

/**
 * Implements `LocalDiffusion` behavior in the SDAI ONNX local diffusion feature layer.
 *
 * @author Dmitriy Moroz
 */
internal class LocalDiffusionImpl(
    /**
     * Exposes the `uNet` value used by the SDAI ONNX local diffusion feature layer.
     *
     * @author Dmitriy Moroz
     */
    private val uNet: UNet,
    /**
     * Exposes the `tokenizer` value used by the SDAI ONNX local diffusion feature layer.
     *
     * @author Dmitriy Moroz
     */
    private val tokenizer: LocalDiffusionTextTokenizer,
    /**
     * Exposes the `ortEnvironmentProvider` value used by the SDAI ONNX local diffusion feature layer.
     *
     * @author Dmitriy Moroz
     */
    private val ortEnvironmentProvider: OrtEnvironmentProvider,
    /**
     * Encodes the Img2Img input image into latents before the UNet runs.
     */
    private val vaeEncoder: VaeEncoder,
    private val vaeEncoderModelLocator: VaeEncoderModelLocator,
    private val vaeEncoderDownloader: VaeEncoderDownloader,
) : LocalDiffusion {

    /**
     * Exposes the `statusFlow` value used by the SDAI ONNX local diffusion feature layer.
     *
     * @author Dmitriy Moroz
     */
    private val statusFlow = MutableSharedFlow<LocalDiffusionStatus>(extraBufferCapacity = 64)

    /**
     * Executes the `process` step in the SDAI ONNX local diffusion feature layer.
     *
     * @param payload generation payload used by the operation.
     * @return Result produced by `process`.
     * @author Dmitriy Moroz
     */
    override suspend fun process(payload: TextToImagePayload): Bitmap = runDiffusion(
        prompt = payload.prompt,
        negativePrompt = payload.negativePrompt,
        samplingSteps = payload.samplingSteps,
        cfgScale = payload.cfgScale,
        seed = payload.seed,
        width = payload.width,
        height = payload.height,
    )

    override val vaeEncoderSizeLabel: String
        get() = VaeEncoderDistribution.SIZE_LABEL

    override fun isImageToImageReady(): Boolean = vaeEncoderModelLocator.isAvailable()

    override fun downloadVaeEncoder(): Flow<DownloadState> = vaeEncoderDownloader.download()

    /**
     * Img2Img: encodes the input image, noises it to the point implied by the denoising strength,
     * then runs the remaining denoising steps.
     *
     * @param payload Img2Img generation payload.
     */
    override suspend fun processImageToImage(payload: ImageToImagePayload): Bitmap {
        val initImage = decodeInitImage(payload.base64Image)
        try {
            return runDiffusion(
                prompt = payload.prompt,
                negativePrompt = payload.negativePrompt,
                samplingSteps = payload.samplingSteps,
                cfgScale = payload.cfgScale,
                seed = payload.seed,
                width = payload.width,
                height = payload.height,
                initImage = initImage,
                strength = payload.denoisingStrength,
            )
        } finally {
            initImage.recycle()
        }
    }

    private suspend fun runDiffusion(
        prompt: String,
        negativePrompt: String,
        samplingSteps: Int,
        cfgScale: Float,
        seed: String,
        width: Int,
        height: Int,
        initImage: Bitmap? = null,
        strength: Float = 1f,
    ): Bitmap = suspendCancellableCoroutine { continuation ->
        try {
            continuation.invokeOnCancellation {
                debugLog(TAG, "{$TAG} Received cancelable signal.")
                interruptGeneration()
            }
            uNet.setCallback(object : UNet.Callback {
                override fun onStep(maxStep: Int, step: Int) {
                    debugLog(TAG, "Received step update: ${maxStep}/${step}")
                    statusFlow.tryEmit(LocalDiffusionStatus(step, maxStep))
                }

                override fun onBuildImage(status: Int, bitmap: Bitmap?) {
                    if (!continuation.isActive) return
                    if (bitmap != null) {
                        debugLog("{$TAG} Bitmap built successfully!")
                        continuation.resume(bitmap)
                    } else {
                        val t = Throwable("Bitmap is null")
                        errorLog(t, "{$TAG} Bitmap is null.")
                        continuation.resumeWithException(t)
                    }
                }
            })

            tokenizer.initialize()
            val batchSize = 1

            val numInferenceSteps = samplingSteps
            val guidanceScale = cfgScale.toDouble()

            val textTokenized = tokenizer.encode(prompt)
            val negTokenized = tokenizer.createUnconditionalInput(negativePrompt)

            val textPromptEmbeddings = tokenizer.tensor(textTokenized)
            val unConditionalEmbedding = tokenizer.tensor(negTokenized)
            val textEmbeddingArray = Array(2) {
                Array(tokenizer.maxLength) {
                    FloatArray(768)
                }
            }

            val textPromptEmbeddingArray = textPromptEmbeddings!!.floatBuffer.array()
            val unConditionalEmbeddingArray = unConditionalEmbedding!!.floatBuffer.array()
            for (i in textPromptEmbeddingArray.indices) {
                textEmbeddingArray[0][i / 768][i % 768] = unConditionalEmbeddingArray[i]
                textEmbeddingArray[1][i / 768][i % 768] = textPromptEmbeddingArray[i]
            }

            val textEmbeddings = OnnxTensor.createTensor(ortEnvironmentProvider.get(), textEmbeddingArray)
            tokenizer.close()

            // Encode before the UNet session exists, so the two never share memory (like the decoder).
            val initLatents = initImage?.let { vaeEncoder.encode(it, width, height) }

            uNet.initialize()
            uNet.inference(
                seedNum = seed.toLongOrNull() ?: 0L,
                numInferenceSteps = numInferenceSteps,
                textEmbeddings = textEmbeddings,
                guidanceScale = guidanceScale,
                batchSize = batchSize,
                width = width,
                height = height,
                initLatents = initLatents,
                strength = strength,
            )
        } catch (e: Exception) {
            errorLog(e, "{$TAG} Caught exception while Local Diffusion process.")
            interruptGeneration()
            if (continuation.isActive) continuation.resumeWithException(e)
        }
    }

    private fun decodeInitImage(base64: String): Bitmap {
        // Tolerate a "data:image/...;base64," prefix.
        val bytes = Base64.decode(base64.substringAfter("base64,"), Base64.DEFAULT)
        return BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
            ?: throw IllegalArgumentException("Unable to decode the input image for Img2Img.")
    }

    // ToDo review method of LocalDiffusion cancellation, now next generation crashes using this approach
    /**
     * Performs the SDAI side effect handled by `interrupt`.
     *
     * @author Dmitriy Moroz
     */
    override suspend fun interrupt() {
        interruptGeneration()
    }

    /**
     * Loads SDAI data through `observeStatus`.
     *
     * @author Dmitriy Moroz
     */
    override fun observeStatus() = statusFlow.asSharedFlow()

    /**
     * Performs the SDAI side effect handled by `interruptGeneration`.
     *
     * @author Dmitriy Moroz
     */
    private fun interruptGeneration() {
        debugLog("{$TAG} Trying to interrupt generation.")
        tokenizer.close()
        uNet.close()
        debugLog("{$TAG} Generation interrupt successful!")
    }
}
