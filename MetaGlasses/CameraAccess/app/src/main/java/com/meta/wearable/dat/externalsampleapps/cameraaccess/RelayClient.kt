package com.meta.wearable.dat.externalsampleapps.cameraaccess

//dependencies
import android.graphics.Bitmap
import android.util.Log
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import okio.ByteString.Companion.toByteString
import org.json.JSONObject
import java.io.ByteArrayOutputStream

// holds a open websocker connection to the
// relay server, compresses images and sends each frames to it
class RelayClient(
    private val url: String,
    private val onPhrase: (String) -> Unit,
) {
    private val client = OkHttpClient()
    private var ws: WebSocket? = null
    @Volatile private var inFlight = false

    // open websocker connection
    fun connect() {
        ws = client.newWebSocket(
            Request.Builder().url(url).build(),
            object : WebSocketListener() {
                override fun onOpen(webSocket: WebSocket, response: Response) {
                    Log.i("relay", "connected to $url")
                }

                override fun onMessage(webSocket: WebSocket, text: String) {
                    inFlight = false
                    val phrase = JSONObject(text).optString("phrase")
                    if (phrase.isNotEmpty()) onPhrase(phrase)
                }

                override fun onFailure(
                    webSocket: WebSocket,
                    t: Throwable,
                    response: Response?,
                ) {
                    Log.e("relay", "failed", t)
                    inFlight = false
                }
            },
        )
    }

    

    // function to compress and send a frame
    fun sendFrame(bitmap: Bitmap) {
        if (inFlight) return
        val out = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.JPEG, 70, out)
        inFlight = true
        ws?.send(out.toByteArray().toByteString())
    }

    fun sendPacket(data: ByteArray) {
        ws?.send(data.toByteString())
    }

    // close the connection
    fun close() {
        ws?.close(1000, null)
    }
}

object Relay {
    var client: RelayClient? = null
}