from gazette.application.ingest import IngestCorpus, read_manifest
from gazette.infrastructure.bm25_index import BM25Index
from gazette.infrastructure.embedders import HashedTfidfEmbedder
from gazette.infrastructure.json_corpus import JsonCorpusSource
from gazette.infrastructure.vector_index import NumpyVectorIndex
from tests.helpers import write_circular


def test_ingest_chunks_one_paragraph_per_chunk_keyed_by_circular_and_para(tmp_path):
    corpus_dir, index_dir = tmp_path / "circulars", tmp_path / "index"
    write_circular(corpus_dir, "GZ-A-2020-01", "2020-01-01", [
        "Cash Transaction Reports shall be filed for cash deposits above the threshold.",
        "Suspicious Transaction Reports shall be filed within seven working days.",
    ])
    write_circular(corpus_dir, "GZ-B-2021-01", "2021-01-01", [
        "Video-based Customer Identification Process requires live customer consent.",
    ])

    report = IngestCorpus(
        JsonCorpusSource(corpus_dir), BM25Index(), NumpyVectorIndex(HashedTfidfEmbedder())
    ).run(index_dir)

    assert (report.circulars, report.paragraphs) == (2, 3)
    assert read_manifest(index_dir) == report

    keyword = BM25Index()
    keyword.load(index_dir)
    vector = NumpyVectorIndex(HashedTfidfEmbedder())
    vector.load(index_dir)
    assert keyword.search("suspicious transaction report", 1)[0].chunk_id == "GZ-A-2020-01#2"
    assert vector.search("video customer identification consent", 1)[0].chunk_id == "GZ-B-2021-01#1"
