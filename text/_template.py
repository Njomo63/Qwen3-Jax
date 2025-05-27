"""Conversation templates."""

import dataclasses
import textwrap


PROMPT = """\
<start_of_turn>user
{}<end_of_turn>
<start_of_turn>model
"""
ANSWER = '{}<end_of_turn>'


@dataclasses.dataclass(frozen=True)
class Turn:
    """Base class for a turn."""

    text: str

    def __repr__(self):
        # Prettier display for multi-line strings.
        if '\n' in self.text:
            text = textwrap.indent(self.text, prefix='    ')
            text = f'"""\n{text}\n"""'
        else:
            text = repr(self.text)

        return f'<{type(self).__name__}({text})>'


@dataclasses.dataclass(frozen=True, repr=False)
class ModelTurn(Turn):
  """Model turn."""


@dataclasses.dataclass(frozen=True, repr=False)
class UserTurn(Turn):
  """User turn."""

