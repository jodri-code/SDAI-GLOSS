package com.shifthackz.aisdv1.data.repository

import com.shifthackz.aisdv1.core.imageprocessing.BitmapToBase64Converter
import com.shifthackz.aisdv1.data.core.CoreGenerationRepository
import com.shifthackz.aisdv1.data.mappers.mapLocalDiffusionImageToImageResult
import com.shifthackz.aisdv1.data.mappers.mapLocalDiffusionToAiGenResult
import com.shifthackz.aisdv1.data.mappers.withModelName
import com.shifthackz.aisdv1.domain.datasource.DownloadableModelDataSource
import com.shifthackz.aisdv1.domain.datasource.GenerationResultDataSource
import com.shifthackz.aisdv1.domain.entity.ImageToImagePayload
import kotlinx.coroutines.flow.Flow
import com.shifthackz.aisdv1.domain.entity.DownloadState
import com.shifthackz.aisdv1.domain.entity.TextToImagePayload
import com.shifthackz.aisdv1.domain.feature.diffusion.LocalDiffusion
import com.shifthackz.aisdv1.domain.feature.work.BackgroundWorkObserver
import com.shifthackz.aisdv1.domain.gateway.MediaStoreGateway
import com.shifthackz.aisdv1.domain.preference.PreferenceManager
import com.shifthackz.aisdv1.domain.repository.LocalDiffusionGenerationRepository
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Implements `LocalDiffusionGenerationRepository` behavior in the SDAI data layer.
 *
 * @author Dmitriy Moroz
 */
internal class LocalDiffusionGenerationRepositoryImpl(
    mediaStoreGateway: MediaStoreGateway,
    localDataSource: GenerationResultDataSource.Local,
    backgroundWorkObserver: BackgroundWorkObserver,
    /**
     * Exposes the `preferenceManager` value used by the SDAI data layer.
     *
     * @author Dmitriy Moroz
     */
    private val preferenceManager: PreferenceManager,
    /**
     * Exposes the `localDiffusion` value used by the SDAI data layer.
     *
     * @author Dmitriy Moroz
     */
    private val localDiffusion: LocalDiffusion,
    /**
     * Exposes the `downloadableLocalDataSource` value used by the SDAI data layer.
     *
     * @author Dmitriy Moroz
     */
    private val downloadableLocalDataSource: DownloadableModelDataSource.Local,
    /**
     * Exposes the `bitmapToBase64Converter` value used by the SDAI data layer.
     *
     * @author Dmitriy Moroz
     */
    private val bitmapToBase64Converter: BitmapToBase64Converter,
) : CoreGenerationRepository(
    mediaStoreGateway = mediaStoreGateway,
    localDataSource = localDataSource,
    preferenceManager = preferenceManager,
    backgroundWorkObserver = backgroundWorkObserver,
), LocalDiffusionGenerationRepository {

    override fun observeStatus() = localDiffusion.observeStatus()

    override suspend fun generateFromText(payload: TextToImagePayload) =
        downloadableLocalDataSource.getSelectedOnnx().let { model ->
            if (model.downloaded) {
                generate(payload, model.name.ifBlank { model.id })
            } else {
                throw IllegalStateException("Model not downloaded.")
            }
        }

    override suspend fun generateFromImage(payload: ImageToImagePayload) =
        downloadableLocalDataSource.getSelectedOnnx().let { model ->
            if (model.downloaded) {
                generateImage(payload, model.name.ifBlank { model.id })
            } else {
                throw IllegalStateException("Model not downloaded.")
            }
        }

    override val vaeEncoderSizeLabel: String
        get() = localDiffusion.vaeEncoderSizeLabel

    override suspend fun isImageToImageReady(): Boolean = localDiffusion.isImageToImageReady()

    override fun downloadVaeEncoder(): Flow<DownloadState> = localDiffusion.downloadVaeEncoder()

    override suspend fun interruptGeneration() = localDiffusion.interrupt()

    private suspend fun generate(
        payload: TextToImagePayload,
        modelName: String,
    ) = withContext(Dispatchers.Default) {
        localDiffusion
            .process(payload)
            .let(BitmapToBase64Converter::Input)
            .let(bitmapToBase64Converter::invoke)
            .base64ImageString
            .let { base64 -> payload to base64 }
            .mapLocalDiffusionToAiGenResult()
            .withModelName(modelName)
    }
        .let { result -> insertGenerationResult(result) }

    private suspend fun generateImage(
        payload: ImageToImagePayload,
        modelName: String,
    ) = withContext(Dispatchers.Default) {
        localDiffusion
            .processImageToImage(payload)
            .let(BitmapToBase64Converter::Input)
            .let(bitmapToBase64Converter::invoke)
            .base64ImageString
            .let { base64 -> payload to base64 }
            .mapLocalDiffusionImageToImageResult()
            .withModelName(modelName)
    }
        .let { result -> insertGenerationResult(result) }
}
