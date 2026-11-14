package com.shifthackz.aisdv1.domain.feature.diffusion

import android.graphics.Bitmap
import com.shifthackz.aisdv1.domain.entity.DownloadState
import com.shifthackz.aisdv1.domain.entity.ImageToImagePayload
import com.shifthackz.aisdv1.domain.entity.LocalDiffusionStatus
import com.shifthackz.aisdv1.domain.entity.TextToImagePayload
import kotlinx.coroutines.flow.Flow

/**
 * Defines the `LocalDiffusion` contract for the SDAI domain layer.
 *
 * @author Dmitriy Moroz
 */
interface LocalDiffusion {
    /**
     * Executes the `process` step in the SDAI domain layer.
     *
     * @param payload generation payload used by the operation.
     * @return Result produced by `process`.
     * @author Dmitriy Moroz
     */
    suspend fun process(payload: TextToImagePayload): Bitmap

    /**
     * Img2Img. Named differently from [process] on purpose: an overload would make
     * `process(any())` ambiguous in existing mocks.
     */
    suspend fun processImageToImage(payload: ImageToImagePayload): Bitmap

    /** Download size of the shared Img2Img VAE encoder, for display. */
    val vaeEncoderSizeLabel: String

    /** Whether the VAE encoder Img2Img needs is available for the selected model. */
    fun isImageToImageReady(): Boolean

    /** Downloads the shared VAE encoder once. */
    fun downloadVaeEncoder(): Flow<DownloadState>
    /**
     * Performs the SDAI side effect handled by `interrupt`.
     *
     * @author Dmitriy Moroz
     */
    suspend fun interrupt()
    /**
     * Loads SDAI data through `observeStatus`.
     *
     * @return Result produced by `observeStatus`.
     * @author Dmitriy Moroz
     */
    fun observeStatus(): Flow<LocalDiffusionStatus>
}
