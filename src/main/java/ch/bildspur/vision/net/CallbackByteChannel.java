package ch.bildspur.vision.net;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.ReadableByteChannel;

class CallbackByteChannel implements ReadableByteChannel {
    NetworkUtility.ProgressCallBack delegate;
    long size;
    ReadableByteChannel rbc;
    long sizeRead;

    CallbackByteChannel(ReadableByteChannel rbc, long expectedSize,
                        NetworkUtility.ProgressCallBack delegate) {
        this.delegate = delegate;
        this.size = expectedSize;
        this.rbc = rbc;
    }
    public void close() throws IOException {
        rbc.close();
    }
    public long getReadSoFar() {
        return sizeRead;
    }

    public boolean isOpen() {
        return rbc.isOpen();
    }

    public int read(ByteBuffer bb) throws IOException {
        int n;
        double progress;
        if ((n = rbc.read(bb)) > 0) {
            sizeRead += n;
            progress = size > 0 ? (double) sizeRead / (double) size
                    * 100.0 : -1.0;
            delegate.callback(this, progress);
        }
        return n;
    }
}
