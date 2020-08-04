<?php
$con=mysqli_connect('localhost','root',"",'studid');
if(!$con)
{
    echo "connection error".mysqli_connect_error();
}
else 
   echo "<h3> Database connection success....</h3>";
?>