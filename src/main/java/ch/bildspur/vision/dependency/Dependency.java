package ch.bildspur.vision.dependency;

import ch.bildspur.vision.web.NetworkUtility;

import java.io.IOException;
import java.nio.file.CopyOption;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.concurrent.atomic.AtomicReference;

import static java.nio.file.StandardCopyOption.REPLACE_EXISTING;

public class Dependency {
    private String name;
    private String url;
    private Path path;
    private String tempSuffix = "_tmp";

    public Dependency(String name) {
        this(name, Repository.repositoryRootUrl + name);
    }

    public Dependency(String name, String url) {
        this.name = name;
        this.url = url;
    }

    public boolean resolve() {
        // lazy create path
        this.path = Paths.get(Repository.localStorageDirectory.toString(), this.name);

        // check if is already there
        if (Files.exists(path)) {
            return true;
        }

        // check temp file
        Path tempPath = Paths.get(path.toString() + tempSuffix);
        try {
            Files.deleteIfExists(tempPath);
        } catch (IOException e) {
            System.err.println("Could not delete " + tempPath.toString());
            return false;
        }

        // try to download
        System.out.print("downloading " + name + ": ");
        AtomicReference<Integer> lastProgress = new AtomicReference<>(0);
        NetworkUtility.downloadFile(url, tempPath, (source, p) -> {
            int last = lastProgress.get();
            int progress = Math.round((float) p);
            int delta = progress - last;

            if (delta >= 10) {
                lastProgress.set(progress);
                System.out.print(".");
            }
        });

        // switch name
        try {
            Files.move(tempPath, path, REPLACE_EXISTING);
        } catch (IOException e) {
            System.err.println("Could not move " + tempPath.toString() + " to " + path.toString());
            e.printStackTrace();
            return false;
        }

        System.out.println(" done!");

        // second check after download
        return Files.exists(path);
    }

    public String getName() {
        return name;
    }

    public String getUrl() {
        return url;
    }

    public Path getPath() {
        return path;
    }
}
