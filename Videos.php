<?php
  if(isset($_FILES['file']) && $_FILES['file']['error'] == UPLOAD_ERR_OK && pathinfo($_FILES['file']['name']) == 'wav'){
    $target_dir = "C:/xampp/htdocs/pythonProject/gorisim/Speech/".$_POST['my_data']."/Audio/";
    if (!file_exists($target_dir)) {
      // create folder if it doesn't exist
      mkdir($target_dir, 0777, true);
   }
    $target_file = $target_dir . basename($_FILES['file']['name']);
    
  // Move uploaded file to target directory
  if(move_uploaded_file($_FILES['file']['tmp_name'], $target_file)) {
    echo "The file ". basename( $_FILES["file"]["name"]). " has been uploaded.";
  } else {
    echo "Sorry, there was an error uploading your file.";
  }
   $script_path = "C:/Users/serha/PycharmProjects/pythonProject/venv/Scripts/python.exe C:\Users\serha\PycharmProjects\pythonProject\SpeechRecognitionReworkedSingleFile.py";
   $output = shell_exec($script_path , $_FILES['file']['name']);
   echo $output;
  }
  elseif(isset($_FILES['file']) && $_FILES['file']['error'] == UPLOAD_ERR_OK && pathinfo($_FILES['file']['name']) == 'wav'){
    $target_dir = "C:/xampp/htdocs/bitirme/wholepose/video/".$_POST['my_data'];
    if (!file_exists($target_dir)) {
        // create folder if it doesn't exist
        mkdir($target_dir, 0777, true);
  }
   $script_path = "C:/Users/serha/PycharmProjects/pythonProject/venv/Scripts/python.exe C:\Users\serha\PycharmProjects\bitirme\video_to_word.py";
   $output = shell_exec($script_path);
}
?>