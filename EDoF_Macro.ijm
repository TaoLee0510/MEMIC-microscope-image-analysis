//── Macro: batch_EDoF.ijm ──────────────────────────────────────────────────────
// 1. Configuration: modify these two lines to set your input/output root directories (trailing '/' required)
inputRoot  = "/Users/4482173/Library/CloudStorage/OneDrive-MoffittCancerCenter/GitHub/ai-coding-integration-plan/week3/MEMIC_output/r01c01/";
outputRoot = "/Users/4482173/Library/CloudStorage/OneDrive-MoffittCancerCenter/GitHub/ai-coding-integration-plan/week3/MEMIC_output/EDoF/r01c01/";

// Ensure the output root exists (create if needed)
if (!File.exists(outputRoot)) {
    File.makeDirectory(outputRoot);
}


// Start recursive processing
processFolder(inputRoot, outputRoot);

function processFolder(inDir, outDir) {
    // Ensure output directory exists
    File.makeDirectory(outDir);
    // List all entries in the current directory
    list = getFileList(inDir);
    for (i = 0; i < list.length; i++) {
        name   = list[i];
        src    = inDir + name;
        dest   = outDir + name;
        if (endsWith(name, ".tif") || endsWith(name, ".tiff")) {
            // TIFF file: process
            // Open the TIFF file
            open(src);
            // Run the EDF Expert plugin
            run("EDF Expert ", " ");
            // Get the result window title (the plugin output window is usually the last one)
            title = getTitle();
            // Construct the save filename: original name without extension + "_EDoF.tiff"
            dot      = lastIndexOf(name, ".");
            baseName = substring(name, 0, dot);
            saveName = baseName + "_EDoF.tiff";
            // Ensure the output subdirectory exists
            File.makeDirectory(outDir);
            // Save and close the images
            selectImage("Output");
            saveAs("Tiff", outDir + saveName);
            close;
            close;
        }
        else {
            // Treat as directory: recurse
            if (!endsWith(src, "/")) src  += "/";
            if (!endsWith(dest, "/")) dest += "/";
            File.makeDirectory(dest);
            processFolder(src, dest);
        }
    }
}
//──────────────────────────────────────────────────────────────────────────────