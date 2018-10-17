################################################
#       Just a script to convert to where each
#       document is on a seperate line
################################################
import sys

def doc_on_line(filename, outfilename):
    infile = open(filename, 'r', encoding='utf-8')
    outfile = open(outfilename, 'w', encoding='utf-8')

    sentfill = "|SENT|" #seperator between different sentences in same doc
    tupfill = "|TUP|"  #seperator between tuples in the same sentence in the same doc

    prev_docid = "" 
    prev_sentid = ""
    docline = ""
    count = 0
    for line in infile:
        print(count)
        count +=1
        splits = line.split("|")
        docid=splits[0]
        sentid=splits[1]
        svo= "|".join(splits[2:5]) 
        sent=splits[5].strip()
        if docid == prev_docid: #still on the same document
            if sentid == prev_sentid: #still on the same sentence, same doc
                docline += tupfill + svo
            else: #same doc, new sentence
                docline +=  sentfill + docid + "|" + sentid + "|" + sent + tupfill + svo
                prev_sentid = sentid
        elif not prev_docid: #if we are on the first line
            prev_docid = docid
            prev_sentid = sentid
            docline =  docid + "|" + sentid + "|" + sent + tupfill + svo
        else: #on to a new document! print what we currently have
            prev_docid = docid
            prev_sentid = sentid
            outfile.write(docline + "\n")
            docline =  docid + "|" + sentid + "|" + sent + tupfill + svo
    infile.close()
    outfile.close()

if __name__ == "__main__":
    infile = sys.argv[1]
    outfile = sys.argv[2]
    doc_on_line(infile, outfile)
