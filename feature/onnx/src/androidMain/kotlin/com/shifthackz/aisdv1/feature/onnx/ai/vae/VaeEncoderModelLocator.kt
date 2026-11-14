package com.shifthackz.aisdv1.feature.onnx.ai.vae

import android.content.Context
import com.shifthackz.aisdv1.core.common.file.FileProviderDescriptor
import com.shifthackz.aisdv1.domain.preference.PreferenceManager
import com.shifthackz.aisdv1.feature.onnx.LocalDiffusionContract.VAE_ENCODER_MODEL
import com.shifthackz.aisdv1.feature.onnx.environment.LocalModelIdProvider
import com.shifthackz.aisdv1.feature.onnx.extensions.modelPathPrefix
import java.io.File

/**
 * Finds the VAE encoder used by Img2Img, in this order:
 *
 * 1. `<selected model folder>/vae_encoder/model.ort` - a per-model override (also what custom
 *    model folders can ship).
 * 2. The shared encoder downloaded once on first use ([sharedFile]).
 */
internal class VaeEncoderModelLocator(
    private val context: Context,
    private val preferenceManager: PreferenceManager,
    private val fileProviderDescriptor: FileProviderDescriptor,
    private val localModelIdProvider: LocalModelIdProvider,
) {

    /** Where the one-time download is stored. Shared by every local model. */
    val sharedFile: File
        get() = File(context.filesDir, "shared/vae_encoder/${VaeEncoderDistribution.VERSION}/model.ort")

    fun isAvailable(): Boolean = findExisting() != null

    fun resolve(): String = findExisting()?.absolutePath
        ?: throw IllegalStateException(
            "Img2Img needs the image encoder, which has not been downloaded yet.",
        )

    private fun findExisting(): File? {
        val modelFolder = modelPathPrefix(preferenceManager, fileProviderDescriptor, localModelIdProvider)
        val perModel = File("$modelFolder/$VAE_ENCODER_MODEL")
        if (perModel.isFile && perModel.length() > 0L) return perModel
        return sharedFile.takeIf { it.isFile && it.length() > 0L }
    }
}
