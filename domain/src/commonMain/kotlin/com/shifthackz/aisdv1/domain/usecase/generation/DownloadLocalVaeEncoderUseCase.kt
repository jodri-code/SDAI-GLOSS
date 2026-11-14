package com.shifthackz.aisdv1.domain.usecase.generation

import com.shifthackz.aisdv1.domain.entity.DownloadState
import kotlinx.coroutines.flow.Flow

/**
 * Downloads the shared VAE encoder that local ONNX Img2Img needs, once.
 */
interface DownloadLocalVaeEncoderUseCase {

    /** Human-readable download size for the confirmation dialog. */
    val sizeLabel: String

    operator fun invoke(): Flow<DownloadState>
}
