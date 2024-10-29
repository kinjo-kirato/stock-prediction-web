"""Add StockPrediction table with updated columns

Revision ID: 7cd063a243b1
Revises: 
Create Date: 2024-10-29 08:13:50.741817

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '7cd063a243b1'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('stock_prediction',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('execution_date', sa.Date(), nullable=False),
    sa.Column('ticker', sa.String(length=10), nullable=False),
    sa.Column('model_type', sa.String(length=50), nullable=False),
    sa.Column('current_price', sa.Float(), nullable=False),
    sa.Column('prediction_date', sa.Date(), nullable=False),
    sa.Column('predicted_price', sa.Float(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('stock_prediction')
    # ### end Alembic commands ###
