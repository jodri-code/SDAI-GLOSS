package com.shifthackz.aisdv1.domain.usecase.generation

/**
 * Whether local ONNX Img2Img has everything it needs on this device (the VAE encoder).
 */
interface IsLocalImageToImageReadyUseCase {
    suspend operator fun invoke(): Boolean
}
