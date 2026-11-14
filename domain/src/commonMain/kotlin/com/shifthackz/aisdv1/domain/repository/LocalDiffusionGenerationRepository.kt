package com.shifthackz.aisdv1.domain.repository

import com.shifthackz.aisdv1.domain.entity.AiGenerationResult
import com.shifthackz.aisdv1.domain.entity.DownloadState
import com.shifthackz.aisdv1.domain.entity.ImageToImagePayload
import com.shifthackz.aisdv1.domain.entity.LocalDiffusionStatus
import com.shifthackz.aisdv1.domain.entity.TextToImagePayload
import kotlinx.coroutines.flow.Flow

/**
 * Defines the `LocalDiffusionGenerationRepository` contract for the SDAI domain layer.
 *
 * @author Dmitriy Moroz
 */
interface LocalDiffusionGenerationRepository {
    /**
     * Loads SDAI data through `observeStatus`.
     *
     * @return Result produced by `observeStatus`.
     * @author Dmitriy Moroz
     */
    fun observeStatus(): Flow<LocalDiffusionStatus>
    /**
     * Executes the `generateFromText` step in the SDAI domain layer.
     *
     * @param payload generation payload used by the operation.
     * @return Result produced by `generateFromText`.
     * @author Dmitriy Moroz
     */
    suspend fun generateFromText(payload: TextToImagePayload): AiGenerationResult
    suspend fun generateFromImage(payload: ImageToImagePayload): AiGenerationResult

    /** Download size of the shared Img2Img VAE encoder, for display. */
    val vaeEncoderSizeLabel: String

    /** Whether local Img2Img has the VAE encoder it needs for the selected model. */
    suspend fun isImageToImageReady(): Boolean

    /** Downloads the shared VAE encoder once. */
    fun downloadVaeEncoder(): Flow<DownloadState>
    /**
     * Performs the SDAI side effect handled by `interruptGeneration`.
     *
     * @author Dmitriy Moroz
     */
    suspend fun interruptGeneration()
}
