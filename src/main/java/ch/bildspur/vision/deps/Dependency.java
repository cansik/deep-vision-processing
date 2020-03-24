package ch.bildspur.vision.deps;

import ch.bildspur.vision.web.NetworkUtility;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.concurrent.atomic.AtomicReference;

public class Dependency {
    private String name;
    private String url;
    private Path path;

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

        // try to download
        System.out.print("downloading: ");

        AtomicReference<Integer> lastProgress = new AtomicReference<>(0);
        NetworkUtility.downloadFile(url, path, (source, progress) -> {
            double p = progress * 100.0;
            int current = lastProgress.get();

            if (p - current > 30) {
                System.out.print(".");
                System.out.flush();
                lastProgress.set(current + 10);
            }
        });
        System.out.println("done!");

        // second check after download
        if (Files.exists(path)) {
            return true;
        }
        return false;
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
