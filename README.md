<h1> Instructions for using Python on Mac OS systems</h1>
    <h3>
    <a href = "https://www.python.org/downloads/" name = "Python3"> Installing python on your computer </a>
    </h3>
    <dl> Pre-requisites 
        <dt> Understanding how to use the terminal </dt>
        <dt> Understanding what a shell is </dt>
        <dt> Understanding how python is activated </dt>
        <dt> How to install python packages </dt>

<h2> Using pythons virtual environment </h2>
    <h3>
        <a href = "https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment" name = "Using pythons virtual environment"> Reference material 
        </a>
    </h3>
    <p>Creating a virtual environment is an important skill that can translate to any operating systems that supports python. The reason being is that not all packages are availible on select operating systems. By creating a virtual environment, this ensures that the packages are not restricted by system compatibility and can be used across systems that support the python language 
    </p>
    <dl> Creating a virtual environment
        <dt> Generic formula </dt>
            <dd> python3 -m venv "file path" </dd>
            <dd> Initiate python3 then make virtual environment at the file path </dd>
            <dd> Example: <strong> python3 -m venv ~/Desktop/virtualenvironment/</strong>  </dd>
    </dl>
    <dl> Accessing the virtual environment
        <dt> Generic formula </dt>
            <dd> source filelocation/bin/activate </dd>
            <dd> activate the file at said file location </dd>
            <dd> Example: <strong> source ~/Desktop/virtualenvironment/bin/activate </strong>  </dd>
    </dl> 
    <dl> Leaving the virtual environment
        <dt> Within the terminal type: <strong> deactivate </strong>  </dt>

<h2> Using the nd2 to tif conversion script </h2>
    <ol>
        <li> Download nd2_to_tif.py </li>
        <li> Save nd2_to_tif.py to a folder of choice </li>
        <li> Open up the terminal </li>
        <li> Change the file directory to where nd2_to_tif.py is located </li>
        <li> For example: <strong> cd ~/Desktop/folder/ </strong> </li>
        <li> Within the terminal, type in: <strong> python3 nd2_to_tif.py </strong>  </li>
        <li> Check if the tif files were generated </li>
   </ol>


