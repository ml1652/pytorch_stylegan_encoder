Get-ChildItem "C:\Users\Mingrui\Desktop\datasets\StyleGANimge_corp\webimage_alignmentTest" -Filter *.jpg | 
Foreach-Object {
    python `
    C:\Users\Mingrui\Desktop\Github\pytorch_stylegan_encoder\Non-ID_Properties_reconstructor.py `
    $_.FullName `
    --learning_rate 1 --weight_landmarkloss 0.0006 --dlatent_path Non_ID_Model_dlatents.npy --iterations 2500
}
