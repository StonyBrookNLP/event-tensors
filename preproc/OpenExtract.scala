package ollie
import io.Source
import edu.knowitall.ollie.Ollie
import edu.knowitall.tool.parse.MaltParser
import java.net.URL
import edu.knowitall.tool.parse.graph.DependencyGraph
import edu.knowitall.openparse.OpenParse
import java.io.{PrintWriter, StringWriter, File}

object OpenExtract{

  def main(args:Array[String]) {

    var inputDir, outputFile = ""
    inputDir = args(0) //the directory in which the files come from 
    outputFile = args(1) //File to print tuples to

    val writer = new PrintWriter(outputFile, "utf-8")
    val directory = new File(inputDir) 
    val sep = "|"
    val parser = new MaltParser
    val openparse = OpenParse.withDefaultModel(OpenParse.Configuration(confidenceThreshold=0.005, expandExtraction=false))
    val ollie = new Ollie(openparse)
    var years_processed = -1
    var months_processed = -1
    var days_processed = -1

//Start Code Specific to NYT Directory Structure
    directory.listFiles.foreach(year => { 
            val in_year_dir = inputDir + "/" + year.getName()
            val input_year = new File(in_year_dir)
            years_processed = years_processed + 1
        input_year.listFiles.foreach(month => {
                val in_month_dir = in_year_dir + "/" + month.getName()
                val input_month = new File(in_month_dir)
                months_processed = months_processed + 1
            input_month.listFiles.foreach(day => {
                    days_processed = days_processed + 1
                    val in_day_dir = in_month_dir + "/" + day.getName()
                    val input_day = new File(in_day_dir)
                    input_day.listFiles.foreach(infile => { //looking at all documents in directory 
                        val docid = year.getName() + "_" + month.getName() + "_" + day.getName() + "_" + infile.getName()

                        var sentence_id = 0 //current sentence we are on
//End Code Specific to NYT Directory Structure
                        Source.fromFile(infile).getLines().foreach(line => {
                            if(line.length() > 30) { //dont bother processing short sentences
                                val sent = parser.dependencyGraph(line)
                                val instances = ollie.extract(sent)

                                for (instance <- instances) {
                                    
                                    val arg1 = instance.extr.arg1.text
                                    val arg2 = instance.extr.arg2.text
                                    val rel = instance.extr.rel.text

                                    val outstr = docid + sep + sentence_id + sep + arg1 + sep + rel + sep + arg2 + sep + line 
                                    println("Years: %s, Months: %s, Days: %s, Extracted: %s".format(years_processed, months_processed, days_processed, sentence_id))

                                    writer.println(outstr)

                                  } 
                              }
                        //end for iterating over tuples
                        
                        sentence_id += 1

                        }) //end for iterating over each line

                        
                      })  //end iterating over files (for one day)

                  }) //days
           })//months
      }) //years 
      writer.close
  }
} 



