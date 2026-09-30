"""Platform-specific DeepStream versions exposed to operators."""

import platform

import click
import pytest
from click.testing import CliRunner

from savant.utils.welcome import get_welcome_message
from scripts import common


@pytest.mark.parametrize(
    ('machine', 'image_name', 'deepstream_version'),
    [
        ('x86_64', 'savant-deepstream', '9.1'),
        ('aarch64', 'savant-deepstream-l4t', '7.1'),
    ],
)
def test_module_image_and_banner_match_platform(
    monkeypatch, machine, image_name, deepstream_version
):
    monkeypatch.setattr(platform, 'machine', lambda: machine)
    monkeypatch.setattr(common, 'SAVANT_VERSION', '0.6.1')
    monkeypatch.setattr(common, 'DOCKER_REGISTRY', None)

    @click.command()
    @common.docker_image_option('savant-deepstream')
    def show_image(docker_image):
        click.echo(docker_image)

    result = CliRunner().invoke(show_image)

    assert result.exit_code == 0
    assert result.output.strip() == (f'{image_name}:0.6.1-{deepstream_version}')
    assert f'DeepStream Version {deepstream_version}' in get_welcome_message()


def test_default_module_image_uses_fork_registry(monkeypatch):
    monkeypatch.setattr(platform, 'machine', lambda: 'x86_64')
    monkeypatch.setattr(common, 'SAVANT_VERSION', '0.6.1')

    @click.command()
    @common.docker_image_option('savant-deepstream')
    def show_image(docker_image):
        click.echo(docker_image)

    result = CliRunner().invoke(show_image)

    assert result.exit_code == 0
    assert result.output.strip() == 'ghcr.io/an1310/savant-deepstream:0.6.1-9.1'
