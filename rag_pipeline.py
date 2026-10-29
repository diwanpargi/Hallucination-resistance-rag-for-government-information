
# The whole flow in one function:
# retrieve -> enough context? -> generate -> validate -> return
from retrieval import retrieve
from generation import generate_answer, build_context, NOT_FOUND
from validation import enough_context, validate_answer
import config


# def answer_question(client, model, gemini, question, top_k=None, threshold=None, with_vectors=False):
#     hits, top_score = retrieve(client, model, question, top_k, threshold)

#     # 1) not enough context: refuse, Gemini is not even called
#     ok, reason = enough_context(hits, top_score)
#    #debug
# #     print("Best score:", top_score)
# #     print("Minimum score:", config.MIN_BEST_SCORE)
# #     print("Retrieved chunks:", len(hits))
# #     print("Context check:", ok, reason)
# # #

#     if not ok:
#         return {"question": question, "answer": NOT_FOUND, "hits": hits,
#                 "top_score": top_score, "reason": reason}

#     # #
#     # context = build_context(hits)
#     # print("Context characters:", len(context))
#     # print("Context preview:", context[:1500])
#     # #

def answer_question(client, model, gemini, question, top_k=None, threshold=None,
                    with_vectors=False):
    hits, top_score = retrieve(client, model, question, top_k, threshold, with_vectors)
 
    # 1) not enough context: refuse, Gemini is not even called
    ok, reason = enough_context(hits, top_score)
    if not ok:
        return {"question": question, "answer": NOT_FOUND, "hits": hits,
                "top_score": top_score, "reason": reason}
 
    # 2) generate the answer
    answer = generate_answer(gemini, build_context(hits), question)
 
    if NOT_FOUND.lower() in answer.lower():
        reason = "model_said_not_found"
    else:
        # 3) citations + claims + scheme mixing. If anything fails, send the fixed refusal.
        ok, reason = validate_answer(gemini, answer, hits)
        print("Validation:", ok, reason)
        if not ok:
            answer = NOT_FOUND
 
    return {"question": question, "answer": answer, "hits": hits,
            "top_score": top_score, "reason": reason
            }

    # 2) generate the answer
    answer = generate_answer(gemini, build_context(hits), question)
    # #
    # print("Gemini answer:", answer)
    # #
    if NOT_FOUND.lower() in answer.lower():
        reason = "model_said_not_found"
    else:
        # 3) citations + claims + scheme mixing. If anything fails, send the fixed refusal.
        ok, reason = validate_answer(gemini, answer, hits)
        print("Validation:", ok, reason)
        if not ok:
            answer = NOT_FOUND

    return {"question": question, "answer": answer, "hits": hits,
            "top_score": top_score, "reason": reason}
