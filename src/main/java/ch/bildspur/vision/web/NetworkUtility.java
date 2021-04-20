package ch.bildspur.vision.web;

import java.io.FileOutputStream;
import java.io.IOException;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.channels.Channels;
import java.nio.channels.ReadableByteChannel;
import java.nio.file.Path;

public class NetworkUtility {
    public static void downloadFile(String remoteURL, Path localPath, ProgressCallBack callback) {
        FileOutputStream fos = null;
        ReadableByteChannel rbc;
        URL url;
        try {
            url = new URL(remoteURL);
            rbc = new CallbackByteChannel(Channels.newChannel(url.openStream()),
                    contentLength(url), callback);
            fos = new FileOutputStream(localPath.toAbsolutePath().toString());
            fos.getChannel().transferFrom(rbc, 0, Long.MAX_VALUE);
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            if (fos != null) {
                try {
                    fos.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    private static int contentLength(URL url) {
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
