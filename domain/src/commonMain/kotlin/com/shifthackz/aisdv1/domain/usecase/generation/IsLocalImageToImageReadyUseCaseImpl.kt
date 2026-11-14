package com.shifthackz.aisdv1.domain.usecase.generation

import com.shifthackz.aisdv1.domain.repository.LocalDiffusionGenerationRepository

internal class IsLocalImageToImageReadyUseCaseImpl(
    private val localDiffusionGenerationRepository: LocalDiffusionGenerationRepository,
) : IsLocalImageToImageReadyUseCase {

    override suspend fun invoke(): Boolean = localDiffusionGenerationRepository.isImageToImageReady()
}
