package com.shifthackz.aisdv1.domain.usecase.generation

import com.shifthackz.aisdv1.domain.entity.DownloadState
import com.shifthackz.aisdv1.domain.repository.LocalDiffusionGenerationRepository
import kotlinx.coroutines.flow.Flow

internal class DownloadLocalVaeEncoderUseCaseImpl(
    private val localDiffusionGenerationRepository: LocalDiffusionGenerationRepository,
) : DownloadLocalVaeEncoderUseCase {

    override val sizeLabel: String
        get() = localDiffusionGenerationRepository.vaeEncoderSizeLabel

    override fun invoke(): Flow<DownloadState> = localDiffusionGenerationRepository.downloadVaeEncoder()
}
