#!/usr/bin/env python3
"""Tests des convertisseurs de datasets — aucun réseau, aucun fichier du dépôt.

    python3 tools/test_dataset_tools.py

Toutes les images sont fabriquées ici, à la main : la suite doit tourner sur une
machine qui n'a ni CIFAR, ni ImageNet, ni internet. Ce qu'elle vérifie n'est pas
« le script s'exécute » mais les propriétés dont dépend un entraînement — les
géométries, l'orientation EXIF, le déterminisme, et le fait qu'un aller-retour
rende les pixels de départ.
"""

import os
import shutil
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json

import batraw
import gen_unet_config
import images_to_raw
import imagenet_to_raw
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def solid(path, size, colour, fmt=None, exif=None):
    """Écrit une image unie. `colour` est un triplet RGB ou un entier (gris)."""
    mode = "L" if isinstance(colour, int) else "RGB"
    image = Image.new(mode, size, colour)
    kwargs = {}
    if exif is not None:
        kwargs["exif"] = exif
    image.save(path, format=fmt, **kwargs)
    return path


class BatrawFormat(unittest.TestCase):
    """Le format lui-même : trois encodages, les mêmes images."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="batlab_batraw_")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_a_round_trip_returns_the_bytes_it_was_given(self):
        """Écrire puis relire rend les octets de départ, sur les trois versions.

        Pour BATRAW1 et 2 le trajet passe par des f32 : c'est donc aussi la
        preuve que la quantisation inverse est exacte, et pas seulement proche.
        """
        source = bytes(range(256)) * 3  # 768 octets = 16×16×3
        for version in (1, 2, 3):
            path = os.path.join(self.dir, f"v{version}.batraw")
            batraw.write(path, source, 1, 16, 16, 3, version=version)
            back, count, w, h, c, got = batraw.read(path)
            self.assertEqual((count, w, h, c, got), (1, 16, 16, 3, version))
            self.assertEqual(bytes(back), source, f"aller-retour cassé en BATRAW{version}")

    def test_the_8_bit_payload_is_a_quarter_of_the_f32_one(self):
        source = bytes(64)
        sizes = {}
        for version in (2, 3):
            path = os.path.join(self.dir, f"size{version}.batraw")
            sizes[version] = batraw.write(path, source, 1, 8, 8, 1, version=version)
        payload = lambda v: sizes[v] - batraw.HEADER_BYTES  # noqa: E731
        self.assertEqual(payload(2), payload(3) * 4)

    def test_the_header_says_what_the_engine_reads(self):
        """L'en-tête est lu par le Rust ; sa disposition est un contrat."""
        path = os.path.join(self.dir, "header.batraw")
        batraw.write(path, bytes(2 * 12), 2, 2, 2, 3, version=3)
        with open(path, "rb") as fh:
            raw = fh.read(batraw.HEADER_BYTES)
        self.assertEqual(raw[:8], b"BATRAW3\0")
        self.assertEqual(struct.unpack("<IIII", raw[8:24]), (2, 2, 2, 3))

    def test_a_payload_that_does_not_match_the_header_is_refused(self):
        path = os.path.join(self.dir, "ragged.batraw")
        with self.assertRaises(ValueError):
            batraw.write(path, bytes(10), 2, 2, 2, 3, version=3)


class ImagesToRaw(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="batlab_images_")
        self.src = os.path.join(self.dir, "src")
        os.makedirs(self.src)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def out(self, name="out.batraw"):
        return os.path.join(self.dir, name)

    def convert(self, **kwargs):
        options = dict(
            root=self.src,
            out_path=self.out(),
            size=8,
            mode="rgb",
            crop="center",
            min_size=0,
            dedup=False,
            version=3,
            verbose=False,
        )
        options.update(kwargs)
        return images_to_raw.convert(**options)

    # -- géométries ------------------------------------------------------

    def test_every_sample_has_the_geometry_that_was_asked_for(self):
        solid(os.path.join(self.src, "a.png"), (100, 40), (200, 10, 10))
        solid(os.path.join(self.src, "b.png"), (7, 250), (10, 200, 10))
        for mode, channels in (("rgb", 3), ("grey", 1)):
            report = self.convert(mode=mode, size=8, out_path=self.out(f"{mode}.batraw"))
            payload, count, w, h, c, _ = batraw.read(self.out(f"{mode}.batraw"))
            self.assertEqual(report["written"], 2)
            self.assertEqual((count, w, h, c), (2, 8, 8, channels))
            self.assertEqual(len(payload), 2 * 8 * 8 * channels)

    def test_a_centred_crop_keeps_the_middle_and_fit_keeps_everything(self):
        """Les deux recadrages doivent être distinguables sur une image conçue pour.

        Une bande rouge verticale au centre d'un fond bleu très large : `center`
        ne doit voir que du rouge, `fit` doit garder du bleu.
        """
        image = Image.new("RGB", (300, 60), (0, 0, 255))
        image.paste(Image.new("RGB", (60, 60), (255, 0, 0)), (120, 0))
        image.save(os.path.join(self.src, "band.png"))

        pixels = {}
        for crop in ("center", "fit"):
            self.convert(crop=crop, size=8, out_path=self.out(f"{crop}.batraw"))
            payload, _, _, _, _, _ = batraw.read(self.out(f"{crop}.batraw"))
            pixels[crop] = bytes(payload)

        centre_is_red = all(pixels["center"][i] > pixels["center"][i + 2] for i in range(0, 192, 3))
        self.assertTrue(centre_is_red, "--crop center a gardé autre chose que le centre")
        self.assertIn(
            True,
            [pixels["fit"][i + 2] > pixels["fit"][i] for i in range(0, 192, 3)],
            "--crop fit a perdu les bords de l'image",
        )

    def test_fit_pads_with_the_neutral_byte(self):
        """Les marges de `fit` doivent être le 0 du modèle, pas du noir.

        Du noir vaut -1 en [-1,1] : c'est un bord dur que le modèle apprendrait
        comme une caractéristique du dataset.
        """
        solid(os.path.join(self.src, "wide.png"), (64, 8), (255, 255, 255))
        self.convert(crop="fit", size=8)
        payload, _, _, _, _, _ = batraw.read(self.out())
        # Ligne du haut : entièrement de la marge.
        self.assertEqual(set(payload[: 8 * 3]), {images_to_raw.NEUTRAL})

    # -- EXIF ------------------------------------------------------------

    def test_the_exif_orientation_tag_is_applied(self):
        """Une photo de téléphone en portrait est stockée en paysage + un tag.

        Sans `exif_transpose`, tout un dataset de photos s'entraîne couché — et
        rien ne le signale. L'image ici est asymétrique haut/bas ; après rotation
        déclarée par le tag, le haut et le bas doivent avoir échangé.
        """
        image = Image.new("RGB", (32, 32), (0, 0, 0))
        image.paste(Image.new("RGB", (32, 16), (255, 255, 255)), (0, 0))  # blanc en haut

        exif = Image.Exif()
        exif[274] = 3  # Orientation = rotation de 180°
        image.save(os.path.join(self.src, "rotated.jpg"), exif=exif)

        self.convert(size=8, out_path=self.out("exif.batraw"))
        payload, _, _, _, _, _ = batraw.read(self.out("exif.batraw"))
        top_left = payload[0]
        bottom_left = payload[(7 * 8) * 3]
        self.assertLess(top_left, 64, "le tag EXIF n'a pas été appliqué (haut resté blanc)")
        self.assertGreater(bottom_left, 192, "le bas aurait dû devenir blanc")

    # -- robustesse ------------------------------------------------------

    def test_a_corrupt_file_is_skipped_and_counted(self):
        solid(os.path.join(self.src, "good.png"), (32, 32), (1, 2, 3))
        with open(os.path.join(self.src, "truncated.jpg"), "wb") as fh:
            fh.write(b"\xff\xd8\xff\xe0" + b"garbage" * 10)
        with open(os.path.join(self.src, "notanimage.png"), "wb") as fh:
            fh.write(b"this is plainly not a png")

        report = self.convert()
        self.assertEqual(report["written"], 1)
        self.assertEqual(report["unreadable"], 2)
        self.assertEqual(report["found"], 3)

    def test_min_size_drops_the_images_that_would_be_mush(self):
        solid(os.path.join(self.src, "big.png"), (64, 64), (1, 2, 3))
        solid(os.path.join(self.src, "tiny.png"), (20, 15), (4, 5, 6))

        # size=32 : le 20×15 est bien la « bouillie » que --min-size vise.
        loose = self.convert(min_size=0, size=32, out_path=self.out("loose.batraw"))
        self.assertEqual(loose["written"], 2)
        self.assertEqual(loose["upscaled"], 1, "l'agrandissement doit être compté et dit")

        strict = self.convert(min_size=32, size=32, out_path=self.out("strict.batraw"))
        self.assertEqual(strict["written"], 1)
        self.assertEqual(strict["too_small"], 1)

    def test_dedup_compares_the_sample_not_the_file(self):
        """Deux encodages du même cliché sont un seul échantillon.

        C'est le point : dédupliquer sur les octets du fichier laisserait passer
        la paire PNG/JPEG que tout scrape ramène, et le modèle verrait l'image
        deux fois.
        """
        image = Image.new("RGB", (64, 64), (30, 90, 150))
        image.save(os.path.join(self.src, "a.png"))
        image.save(os.path.join(self.src, "b.png"))
        image.resize((128, 128), Image.NEAREST).save(os.path.join(self.src, "c.png"))
        solid(os.path.join(self.src, "d.png"), (64, 64), (200, 10, 10))

        without = self.convert(dedup=False, out_path=self.out("dup.batraw"))
        self.assertEqual(without["written"], 4)

        with_dedup = self.convert(dedup=True, out_path=self.out("dedup.batraw"))
        self.assertEqual(with_dedup["written"], 2)
        self.assertEqual(with_dedup["duplicates"], 2)

    def test_an_empty_or_unusable_folder_is_an_error(self):
        with self.assertRaises(ValueError):
            self.convert()

    # -- déterminisme et aller-retour -----------------------------------

    def test_two_runs_over_the_same_folder_give_the_same_file(self):
        """L'ordre vient du chemin, pas de l'ordre du système de fichiers.

        Créés dans un ordre, nommés dans un autre : si le tri n'était pas fait,
        les deux exécutions différeraient sur cette machine ou sur une autre.
        """
        os.makedirs(os.path.join(self.src, "z"))
        os.makedirs(os.path.join(self.src, "a"))
        solid(os.path.join(self.src, "z", "2.png"), (32, 32), (10, 20, 30))
        solid(os.path.join(self.src, "a", "1.png"), (32, 32), (40, 50, 60))
        solid(os.path.join(self.src, "m.png"), (32, 32), (70, 80, 90))

        first = self.out("first.batraw")
        second = self.out("second.batraw")
        self.convert(out_path=first)
        self.convert(out_path=second)
        with open(first, "rb") as a, open(second, "rb") as b:
            self.assertEqual(a.read(), b.read())

        # …et l'ordre est bien celui des chemins triés : a/1, m, z/2.
        payload, _, _, _, _, _ = batraw.read(first)
        self.assertEqual(payload[0:3], bytes((40, 50, 60)))
        self.assertEqual(payload[192:195], bytes((70, 80, 90)))
        self.assertEqual(payload[384:387], bytes((10, 20, 30)))

    def test_a_flat_colour_survives_the_whole_trip(self):
        """L'aller-retour complet : image → .batraw → pixels.

        Une couleur unie traverse le recadrage et le redimensionnement sans
        changer, donc tout écart est un défaut du pipeline et pas du filtre de
        rééchantillonnage. Vérifié sur les trois formats : c'est aussi la preuve
        que le choix de `--format` ne change pas les images.
        """
        solid(os.path.join(self.src, "flat.png"), (40, 40), (17, 200, 99))
        for version in (1, 2, 3):
            path = self.out(f"trip{version}.batraw")
            self.convert(version=version, out_path=path)
            payload, count, w, h, c, _ = batraw.read(path)
            self.assertEqual((count, w, h, c), (1, 8, 8, 3))
            self.assertEqual(set(payload[0::3]), {17})
            self.assertEqual(set(payload[1::3]), {200})
            self.assertEqual(set(payload[2::3]), {99})

    def test_grey_uses_the_same_luminance_as_the_cifar_converter(self):
        """Les deux convertisseurs doivent produire le même gris.

        Sinon un fine-tune de CIFAR vers des images à soi change la définition
        de « gris » en cours de route.
        """
        solid(os.path.join(self.src, "colour.png"), (32, 32), (200, 100, 50))
        self.convert(mode="grey", out_path=self.out("grey.batraw"))
        payload, _, _, _, _, _ = batraw.read(self.out("grey.batraw"))
        expected = round(0.299 * 200 + 0.587 * 100 + 0.114 * 50)
        self.assertLessEqual(
            abs(payload[0] - expected), 1, f"gris {payload[0]}, BT.601 donne {expected}"
        )

    # -- planche ---------------------------------------------------------

    def test_the_contact_sheet_shows_the_whole_dataset_not_its_head(self):
        """La planche échantillonne tout le dataset.

        Un scrape trié par chemin a ses 64 premiers fichiers dans le même
        sous-dossier ; une planche qui ne montrerait qu'eux mentirait par
        omission. Ici la dernière image est d'une couleur unique — elle doit
        apparaître.
        """
        for i in range(200):
            solid(os.path.join(self.src, f"{i:03d}.png"), (32, 32), (i, i, i))
        solid(os.path.join(self.src, "zzz.png"), (32, 32), (255, 0, 0))

        self.convert(size=8)
        png = os.path.join(self.dir, "sheet", "plate.png")
        shown = images_to_raw.contact_sheet(self.out(), png, columns=4, cell=16, rows=4)

        self.assertEqual(shown, 16)
        self.assertTrue(os.path.exists(png))
        sheet = Image.open(png)
        self.assertEqual(sheet.size, (4 * 16, 4 * 16))
        colours = [c for _, c in sheet.convert("RGB").getcolors(4096)]
        self.assertIn((255, 0, 0), colours, "la dernière image du dataset n'est pas sur la planche")
        self.assertIn((0, 0, 0), colours, "la première image du dataset n'est pas sur la planche")

    def test_the_contact_sheet_works_in_greyscale_too(self):
        solid(os.path.join(self.src, "a.png"), (32, 32), (10, 10, 10))
        self.convert(mode="grey", size=8)
        png = os.path.join(self.dir, "grey_sheet.png")
        images_to_raw.contact_sheet(self.out(), png, columns=2, cell=8, rows=2)
        self.assertEqual(Image.open(png).mode, "L")


class Imagenet32ToRaw(unittest.TestCase):
    """L'ingestion ImageNet 32×32, sur des batches fabriqués ici.

    Le vrai jeu n'est pas dans le dépôt et ne le sera jamais (3,94 Go) : ce qui
    est testé est le contrat de lecture, sur des fichiers au même format.
    """

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="batlab_imagenet_")
        self.src = os.path.join(self.dir, "in")
        os.makedirs(self.src)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def write_batch(self, name, count, first=0, keys_as_bytes=False, side=32):
        """Un batch au format de l'archive : pickle, `data` en PLANS (n, 3·côté²).

        `side` par défaut 32 (release historique) ; 64 fabrique une batche de la
        release 64×64, dont seule la longueur de ligne change.
        """
        import pickle

        import numpy as np

        plane = side * side
        rows = []
        for i in range(count):
            value = (first + i) % 256
            rows.append(
                [value] * plane + [(value + 1) % 256] * plane + [(value + 2) % 256] * plane
            )
        data = np.asarray(rows, dtype=np.uint8)
        payload = {"data": data, "labels": list(range(1, count + 1))}
        if keys_as_bytes:
            payload = {k.encode(): v for k, v in payload.items()}
        path = os.path.join(self.src, name)
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)
        return path

    def test_the_three_planes_become_pixels(self):
        """Le piège du format : 3072 octets sont 3 plans, pas des pixels.

        Lu tel quel, le dataset donne des images au tiers haut rouge, tiers
        médian vert, tiers bas bleu — et rien ne lève d'erreur. Chaque image ici
        a une valeur constante par plan, donc chaque PIXEL doit valoir
        (v, v+1, v+2).
        """
        self.write_batch("train_data_batch_1", 2, first=10)
        out = os.path.join(self.dir, "planes.batraw")
        imagenet_to_raw.convert(self.src, out, mode="rgb")

        payload, count, w, h, c = batraw.read(out)[:5]
        self.assertEqual((count, w, h, c), (2, 32, 32, 3))
        first_image = payload[:3072]
        self.assertEqual(set(first_image[0::3]), {10})
        self.assertEqual(set(first_image[1::3]), {11})
        self.assertEqual(set(first_image[2::3]), {12})

    def test_a_64x64_archive_is_detected_from_the_row_length(self):
        """La release 64×64 a le même format ; seule la ligne passe à 12288.

        La taille se déduit de `data.shape[1]`, elle n'est pas demandée. Une
        image à plans constants doit ressortir en pixels (v, v+1, v+2) sur une
        grille 64×64.
        """
        self.write_batch("train_data_batch_1", 1, first=5, side=64)
        out = os.path.join(self.dir, "big.batraw")
        report = imagenet_to_raw.convert(self.src, out, mode="rgb")
        self.assertEqual(report["side"], 64)
        payload, count, w, h, c = batraw.read(out)[:5]
        self.assertEqual((count, w, h, c), (1, 64, 64, 3))
        self.assertEqual((payload[0], payload[1], payload[2]), (5, 6, 7))

    def test_a_row_that_is_not_three_square_planes_is_refused(self):
        """Une longueur qui n'est pas 3·côté² n'est pas un ImageNet — refus net."""
        with self.assertRaises(ValueError):
            imagenet_to_raw.side_from_row(3000)  # 1000 pixels, pas un carré
        self.assertEqual(imagenet_to_raw.side_from_row(3072), 32)
        self.assertEqual(imagenet_to_raw.side_from_row(12288), 64)

    def test_a_folder_may_not_mix_two_sizes(self):
        """Un dossier 32×32 + 64×64 donnerait un .batraw incohérent : refus."""
        self.write_batch("train_data_batch_1", 1, side=32)
        self.write_batch("train_data_batch_2", 1, side=64)
        out = os.path.join(self.dir, "mixed.batraw")
        with self.assertRaises(ValueError):
            imagenet_to_raw.convert(self.src, out, mode="rgb")

    def test_batches_are_read_in_numeric_order(self):
        """`_10` vient après `_2`, pas entre `_1` et `_2`.

        Un tri lexicographique ferait dépendre l'ordre du dataset du nombre de
        fichiers présents.
        """
        self.write_batch("train_data_batch_1", 1, first=0)
        self.write_batch("train_data_batch_2", 1, first=100)
        self.write_batch("train_data_batch_10", 1, first=200)

        out = os.path.join(self.dir, "order.batraw")
        report = imagenet_to_raw.convert(self.src, out, mode="rgb")
        self.assertEqual(report["written"], 3)
        payload = batraw.read(out)[0]
        firsts = [payload[i * 3072] for i in range(3)]
        self.assertEqual(firsts, [0, 100, 200])

    def test_limit_stops_early(self):
        self.write_batch("train_data_batch_1", 5)
        self.write_batch("train_data_batch_2", 5)
        out = os.path.join(self.dir, "small.batraw")
        report = imagenet_to_raw.convert(self.src, out, mode="rgb", limit=7)
        self.assertEqual(report["written"], 7)
        # Le compte de l'en-tête est corrigé au close() : il doit coller.
        self.assertEqual(batraw.read(out)[1], 7)

    def test_keys_may_be_str_or_bytes(self):
        """Selon la façon dont l'archive a été picklée, les clés diffèrent."""
        self.write_batch("train_data_batch_1", 1, keys_as_bytes=True)
        out = os.path.join(self.dir, "bytes_keys.batraw")
        self.assertEqual(imagenet_to_raw.convert(self.src, out)["written"], 1)

    def test_a_missing_folder_says_what_it_expected(self):
        empty = os.path.join(self.dir, "empty")
        os.makedirs(empty)
        with self.assertRaises(FileNotFoundError) as caught:
            imagenet_to_raw.convert(empty, os.path.join(self.dir, "x.batraw"))
        self.assertIn("train_data_batch", str(caught.exception))

    def test_a_batch_of_the_wrong_shape_is_refused(self):
        import pickle

        import numpy as np

        path = os.path.join(self.src, "train_data_batch_1")
        with open(path, "wb") as fh:
            pickle.dump({"data": np.zeros((2, 100), dtype=np.uint8), "labels": [1, 2]}, fh)
        with self.assertRaises(ValueError):
            imagenet_to_raw.convert(self.src, os.path.join(self.dir, "x.batraw"))

    def test_grey_matches_the_cifar_converter(self):
        """Les deux fondations doivent définir « gris » de la même façon."""
        self.write_batch("train_data_batch_1", 1, first=200)
        out = os.path.join(self.dir, "grey.batraw")
        imagenet_to_raw.convert(self.src, out, mode="grey")
        payload, count, w, h, c = batraw.read(out)[:5]
        self.assertEqual((count, w, h, c), (1, 32, 32, 1))
        self.assertEqual(payload[0], round(0.299 * 200 + 0.587 * 201 + 0.114 * 202))

    def test_a_writer_left_with_half_an_image_refuses_to_close(self):
        """L'en-tête d'un `.batraw` ne doit jamais annoncer plus qu'il ne porte."""
        path = os.path.join(self.dir, "partial.batraw")
        writer = batraw.Writer(path, 2, 2, 1)
        writer.append(bytes(6))  # une image et demie
        with self.assertRaises(ValueError):
            writer.close()


class GenUnetConfig(unittest.TestCase):
    """Le générateur de config U-Net : dims dérivées, étages libres."""

    def test_the_default_reproduces_greyscale_L_byte_for_byte(self):
        """Généraliser à N étages ne doit pas bouger d'un octet le défaut.

        Le `config_file` de `Greyscale_Diffusion_L` est la référence : taille 32,
        largeurs 32/64/128, trois étages. La sortie du générateur par défaut doit
        lui être identique — c'est ce qui garde intacts les modèles qui en sont
        nés.
        """
        _, config = gen_unet_config.build_config()
        rendered = json.dumps(config, indent=2) + "\n"
        reference = os.path.join(
            REPO_ROOT, "Models", "Greyscale_Diffusion_L", "config_file")
        with open(reference, "r") as fh:
            self.assertEqual(rendered, fh.read())

    def test_kernel_depth_equals_the_running_channel_count_at_every_layer(self):
        """La raison d'être du script : `dim_kernel.z == dim_input.z` partout.

        Une UpsampleConv dont la profondeur de noyau ment corrompt en silence
        (son shader indexe les poids avec IC = dim_input.z). On le vérifie sur une
        pile profonde à quatre étages, celle que le défaut ne couvre pas.
        """
        _, config = gen_unet_config.build_config(
            size=64, widths=(48, 96, 192, 384), attention=True)
        for layer in config["layers"]:
            for kind in ("Convolution", "UpsampleConv"):
                if kind in layer:
                    spec = layer[kind]
                    self.assertEqual(
                        spec["dim_kernel"][2], spec["dim_input"][2],
                        f"{kind}: kernel depth {spec['dim_kernel'][2]} != "
                        f"input channels {spec['dim_input'][2]}")

    def test_a_size_that_does_not_halve_cleanly_is_refused(self):
        """`size` doit se diviser par 2 autant de fois qu'il y a de descentes."""
        # 20 ne se divise pas par 2**3 = 8 : refusé.
        with self.assertRaises(ValueError):
            gen_unet_config.build_config(size=20, widths=(32, 64, 128, 256))
        # 64 se divise par 8 : accepté.
        gen_unet_config.build_config(size=64, widths=(32, 64, 128, 256))

    def test_the_stack_grows_by_one_encoder_and_one_decoder_per_extra_stage(self):
        """Un étage de plus = une descente + une remontée symétriques de plus."""
        _, three = gen_unet_config.build_config(size=32, widths=(32, 64, 128))
        _, four = gen_unet_config.build_config(size=64, widths=(32, 64, 128, 256))
        downs3 = sum(1 for l in three["layers"]
                     if "Convolution" in l and l["Convolution"]["stride"] == 2)
        downs4 = sum(1 for l in four["layers"]
                     if "Convolution" in l and l["Convolution"]["stride"] == 2)
        ups3 = sum(1 for l in three["layers"] if "UpsampleConv" in l)
        ups4 = sum(1 for l in four["layers"] if "UpsampleConv" in l)
        self.assertEqual((downs3, ups3), (2, 2))
        self.assertEqual((downs4, ups4), (3, 3))


if __name__ == "__main__":
    unittest.main(verbosity=2)
