package ch.bildspur.vision.deps;

import ch.bildspur.vision.net.CallbackByteChannel;
import ch.bildspur.vision.net.NetworkUtility;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

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
        this.path = Paths.get(Repository.localStorageDirectory.toString(),this.name);
    }

    public boolean resolve() {
        // check if is already there
        if(Files.exists(path)) {
            return true;
        }

        // try to download
        NetworkUtility.downloadFile(url, path, (source, progress) -> {
            System.out.print("Progress: " + progress);
        });

        // second check after download
        if(Files.exists(path)) {
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
