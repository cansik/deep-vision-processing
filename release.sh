#!/bin/sh
if [ $# -eq 0 ]
  then
  	echo ""
    echo "Please provide following arguments:"
    echo "$0 [VERSION]"
    exit
fi

# parameter
VERSION=$1
ARCHIVE_NAME="deepvision"
TERMINAL="$(expr substr $(uname -s) 1 6)"
echo "Terminal: $TERMINAL"

echo $PWD

echo clean up...
rm -r -f build

echo compiling...
#|| [ $TERMINAL == "MINGW6" ]
if [ $TERMINAL == "CYGWIN" ] ;then
    echo running gradle commands on windows
    gradlew.bat build
    gradlew.bat fatjar
    gradlew.bat javadoc
else
    echo running gradle commands on unix
    ./gradlew build
    ./gradlew fatjar
    ./gradlew javadoc
fi

echo "copy files..."
OUTPUT="release/$ARCHIVE_NAME"
OUTPUTV="release/$ARCHIVE_NAME_$VERSION"
rm -r -f "$OUTPUT"
rm -r -f "$OUTPUTV"

mkdir -p "$OUTPUT/library"

# copy files
cp -f library.properties "release/$ARCHIVE_NAME.txt"
# cp -a "build/libs/lib/." "$OUTPUT/library/"
cp "build/libs/$ARCHIVE_NAME-complete.jar" "$OUTPUT/library/$ARCHIVE_NAME.jar"
# cp -r native "$OUTPUT/library/"
cp -r "build/docs/javadoc" "$OUTPUT/reference"

# clean networks from examples
cd "examples"
rm -rf **/networks
cd ..

cp -r "examples" "$OUTPUT/"
cp library.properties "$OUTPUT/"
cp -r readme "$OUTPUT/"
cp README.md "$OUTPUT/"
cp -r "src" "$OUTPUT/"

# create release files
cd "release/"
rm -f "$ARCHIVE_NAME.zip"
zip -r "$ARCHIVE_NAME.zip" "$ARCHIVE_NAME" -x "*.DS_Store"

# install
if [[ $2 = "-i" ]]; then
   echo "installing library..."
   rm -rf "~/Documents/Processing/libraries/$ARCHIVE_NAME"
   cp -r "$OUTPUT" "~/Documents/Processing/libraries/"
fi

# store it with version number
cd ..
mv -f "$OUTPUT" "$OUTPUTV"

echo "-------------------------"
echo "finished release $VERSION"
