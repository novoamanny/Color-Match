Install Git LFS to be able to push to repo. It is needed for the models/.onnx file.
```bash
brew install git-lfs 
```
Track .onnx file.
```bash
git lfs track "*.onnx"
```
Track Changes
```bash
git add <filepath> || .
git commit -m "message"
```
Re-add .onnx file
```bash
git rm --cached models/u2net.onnx
git add models/u2net.onnx
git commit -m "Add u2net.onnx via LFS"
```
Push to Repo
```bash
git push origin main
```