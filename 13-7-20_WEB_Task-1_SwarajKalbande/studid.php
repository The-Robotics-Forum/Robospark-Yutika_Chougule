
<?php
include('db.php');
if(isset($_POST['submit']))
{
    $id = $_POST['stud_id'];
    echo $id;
    
    $query = "SELECT * FROM `student` WHERE `stud_id`='$id'";
    $run = mysqli_query($con, $query);
    $row = mysqli_num_rows($run);

    if ($row == 1){
        $current = mysqli_fetch_assoc($run);
        $name = $current['stud_name'];
        $branch = $current['branch'];
        $cgpa = $current['cgpa'];
        echo "
        <!DOCTYPE html>
        <head>
            <title>Document</title>
            <link rel='stylesheet' href='studid.css'>
        </head>
        <body>
        <div class='portal'>
            <h4><em>Student Found</em></h4>
            <p>Student ID is: $id</p>
            <p>Student name is: $name</p>
            <p>Branch: $branch</p>
            <p>CGPA: $cgpa</p>
        </div>
        </body>
        </html>
        ";
    }
    else{
        echo "unknown id entered";
    }
    }
    ?>