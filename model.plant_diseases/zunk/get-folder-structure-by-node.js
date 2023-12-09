const fs = require("fs");
const path = require("path");

function getDirectories(rootDir) {
  const result = [];

  function traverseDir(currentPath) {
    const files = fs.readdirSync(currentPath);

    files.forEach((file) => {
      const fullPath = path.join(currentPath, file);
      const stats = fs.statSync(fullPath);

      if (stats.isDirectory()) {
        result.push({
          path: fullPath,
          name: file,
        });

        // Recursively traverse subdirectories
        traverseDir(fullPath);
      }
    });
  }

  traverseDir(rootDir);
  return result;
}

// Replace 'YOUR_ROOT_DIRECTORY_PATH' with the path to your parent folder
let rootDirectory = "./";
const directoryStructure = getDirectories(rootDirectory);

console.log("\n", directoryStructure);
