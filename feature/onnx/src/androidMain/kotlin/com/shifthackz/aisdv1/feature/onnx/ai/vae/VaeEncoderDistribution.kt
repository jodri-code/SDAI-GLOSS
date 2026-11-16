package com.shifthackz.aisdv1.feature.onnx.ai.vae

/**
 * Where the shared Img2Img VAE encoder is downloaded from.
 *
 * One encoder serves every local ONNX model, so it is a single file, fetched once on first use.
 * Fill in [SOURCES] and [SHA256] after uploading the file; until then the download reports
 * "not configured" instead of failing silently.
 */
internal object VaeEncoderDistribution {

    /**
     * Direct file URLs, tried in order. The first one that downloads AND matches [SHA256] wins,
     * so list the most reliable host first and the mirrors after it.
     */
    val SOURCES: List<String> = listOf(
        // The file name in the URL does not matter; the app always saves it as model.ort.
        "https://huggingface.co/jodri-code/onnx-sd1.5-universal-encoder/resolve/main/model.ort",
        "https://cdn.sooox.cc/DyekWnb8Pl3N.ort",
        // Add mirrors here; they are tried in order if the one above fails or does not match SHA256.
    )

    /** Lowercase hex SHA-256 of the hosted file (`sha256sum vae_encoder/model.ort`). */
    const val SHA256: String = "d831b2a73c39a3695c70a87c0dec510f99882fcd139e8e9086be797c0fa3b2e4"

    /** Shown to the user in the confirmation dialog. */
    const val SIZE_LABEL: String = "137 MB"

    /** Part of the on-device path. Bump it when hosting a different encoder so devices fetch the new one. */
    const val VERSION: String = "v1"
}
