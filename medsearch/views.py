from django.shortcuts import render
from django.http import HttpResponse
from ir.bsbi import BSBIIndex
from ir.compression import VBEPostings
from ir.letor import LETOR

def index(request):
    return render(request, "search.html")

def ranking(request):
    BSBI_instance = BSBIIndex(data_dir = 'ir/collection', \
                          postings_encoding = VBEPostings, \
                          output_dir = 'ir/index')
    LETOR_instance = LETOR(model_dir='ir/model')

    query = request.GET['query']

    docs = []
    for (score, doc) in BSBI_instance.retrieve_bm25(query, k = 10):
        docs.append((doc, " ".join(BSBI_instance.docs[doc]), BSBI_instance.docs_real[doc]))
    
    sorted_did_scores = LETOR_instance.predict(query, docs)

    return render(request, "search.html", {"list":sorted_did_scores})
