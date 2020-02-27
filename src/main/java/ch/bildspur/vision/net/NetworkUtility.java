package ch.bildspur.vision.net;

import java.io.FileOutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.channels.Channels;
import java.nio.channels.ReadableByteChannel;
import java.nio.file.Path;

public class NetworkUtility {
    public void downloadFile(String remoteURL, Path localPath, CallbackByteChannel callback) {
        FileOutputStream fos;
        ReadableByteChannel rbc;
        URL url;
        try {
            url = new URL(remoteURL);
            rbc = new CallbackByteChannel(Channels.newChannel(url.openStream()),
                    contentLength(url), callback.delegate);
            fos = new FileOutputStream(localPath.toAbsolutePath().toString());
            fos.getChannel().transferFrom(rbc, 0, Long.MAX_VALUE);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private int contentLength(URL url) {
        HttpURLConnection connection;
        int contentLength = -1;
        try {
            connection = (HttpURLConnection) url.openConnection();
            contentLength = connection.getContentLength();
        } catch (Exception e) {
            e.printStackTrace();
        }
        return contentLength;
    }
}
