import os
import pathlib
import shutil

import pdp
import keras_autodoc
import tutobooks

PAGES = {
    "AggByCols.md": [
        "pdp.AggByCols",
    ],
}


aliases_needed = [
]


ROOT = "http://autokeras.com/"

pdpipe_dir = pathlib.Path(__file__).resolve().parents[1]


def py_to_nb_md(dest_dir):
    dir_path = "py"
    for file_path in os.listdir("py/"):
        file_name = file_path
        py_path = os.path.join(dir_path, file_path)
        file_name_no_ext = os.path.splitext(file_name)[0]
        ext = os.path.splitext(file_name)[1]

        if ext != ".py":
            continue

        nb_path = os.path.join("ipynb", file_name_no_ext + ".ipynb")
        md_path = os.path.join(dest_dir, "tutorial", file_name_no_ext + ".md")

        tutobooks.py_to_md(py_path, nb_path, md_path, "templates/img")

        github_repo_dir = "keras-team/autokeras/blob/master/docs/"
        with open(md_path, "r") as md_file:
            button_lines = [
                ":material-link: "
                "[**View in Colab**](https://colab.research.google.com/github/"
                + github_repo_dir
                + "ipynb/"
                + file_name_no_ext
                + ".ipynb"
                + ")   &nbsp; &nbsp;"
                # + '<span class="k-dot">•</span>'
                + ":octicons-octoface-16: "
                "[**GitHub source**](https://github.com/"
                + github_repo_dir
                + "py/"
                + file_name_no_ext
                + ".py)",
                "\n",
            ]
            md_content = "".join(button_lines) + "\n" + md_file.read()

        with open(md_path, "w") as md_file:
            md_file.write(md_content)


def generate(dest_dir):
    template_dir = pdpipe_dir / "docs" / "templates"
    doc_generator = keras_autodoc.DocumentationGenerator(
        PAGES,
        "https://github.com/pdpipe/pdpipe/blob/master",
        template_dir,
        pdpipe_dir / "examples",
        extra_aliases=aliases_needed,
    )
    doc_generator.generate(dest_dir)
    readme = (pdpipe_dir / "README.md").read_text()
    index = (template_dir / "index.md").read_text()
    index = index.replace("{{autogenerated}}", readme[readme.find("##"):])
    (dest_dir / "index.md").write_text(index, encoding="utf-8")
    shutil.copyfile(
        pdpipe_dir / ".github" / "CONTRIBUTING.md",
        dest_dir / "contributing.md"
    )

    py_to_nb_md(dest_dir)


if __name__ == "__main__":
    generate(pdpipe_dir / "docs" / "sources")
