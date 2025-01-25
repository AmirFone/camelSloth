# setup.py
import setuptools

setuptools.setup(
    name="camelsloth",
    version="0.1.0",
    description="Meta library to generate CoT with CAMEL and finetune with Unsloth",
    packages=setuptools.find_packages(),
    install_requires=[
        "pyyaml",
        "huggingface-hub",
        "datasets",
        "transformers",
        "trl",
        "unsloth",
        "camel-ai==0.2.16"
    ],
    python_requires=">=3.8"
)
