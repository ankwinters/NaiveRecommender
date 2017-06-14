package main

import (
	"fmt"
	//"io"
	"bufio"
	"os"
	"strings"
	"strconv"
	"io"
)



func check(e error) {
    if e != nil {
        panic(e)
    }
}

func read_file(path string, matrix [][]int){
	matrix = make([][]int, 943)

	// 创建二维Slice
	for i := range matrix {
		var subArray []int
    	subArray = make([]int, 1682)
    	for j := range subArray {
        	subArray[j] = j + 1
    	}
    	matrix[i] = subArray
	}

	// Open file
	train_file, readError := os.Open(path)
	check(readError)
	defer train_file.Close()
	// Read file content to matrix
	reader := bufio.NewReader(train_file)
	for {
		line,err:=reader.ReadString('\n')
		if err == io.EOF {
            break
        }
		check(err)
		info := strings.Split(line, " ")
		user_id,_ := strconv.Atoi(info[0])
		item_id,_ := strconv.Atoi(info[1])
		value,_ := strconv.Atoi(info[2])
		matrix[user_id-1][item_id-1] = value
	}

	return matrix
}

func get_recommendations(){
	return
}


func main()  {
	if len(os.Args)!=2 {
        //fmt.Errorf("error input")
        fmt.Println("error input")
        return
    }
	mat:=read_file(os.Args[1])
	fmt.Println(mat[942][1329])
	
}
