package ch.bildspur.vision.util;

import processing.core.PApplet;
import processing.core.PConstants;

import java.net.URL;

public class ProcessingUtils {
    public static String getLibPath(Object caller) {
        URL url = caller.getClass().getResource("DeepVision.class");
        if (url != null) {
            // Convert URL to string, taking care of spaces represented by the "%20"
            // string.
            String path = url.toString().replace("%20", " ");
            int n0 = path.indexOf('/');

            int n1 = -1;


            n1 = path.indexOf("deepvision.jar");
            if (PApplet.platform == PConstants.WINDOWS) { //platform Windows
                // In Windows, path string starts with "jar file/C:/..."
                // so the substring up to the first / is removed.
                n0++;
            }


            if ((-1 < n0) && (-1 < n1)) {
                return path.substring(n0, n1);
            } else {
                return "";
            }
        }
        return "";
    }
}
