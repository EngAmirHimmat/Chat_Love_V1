# Chat_Love_V1

pip install torch transformers datasets accelerate

كيف يعمل؟
تحميل بيانات المحادثات من ملف CSV.
تحضير النموذج والتوكنايزر من مكتبة Hugging Face.
تحويل النصوص إلى Tokens باستخدام tokenizer.
تقسيم البيانات إلى تدريب واختبار بنسبة 90% تدريب و10% اختبار.
إعداد عملية التدريب باستخدام Trainer.
بدء التدريب وحفظ النموذج النهائي.
كيف تستخدم النموذج بعد التدريب؟
بعد التدريب، يمكنك استخدامه للدردشة عبر:
run.py
