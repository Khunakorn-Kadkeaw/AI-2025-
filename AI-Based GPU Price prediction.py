{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyMN/F9jmr4ytB60H5z0va5U",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Khunakorn-Kadkeaw/AI-2025-/blob/main/AI-Based%20GPU%20Price%20prediction.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "import re\n",
        "from sklearn.preprocessing import LabelEncoder\n",
        "from sklearn.ensemble import RandomForestRegressor"
      ],
      "metadata": {
        "id": "-WcYEgbGscge"
      },
      "execution_count": 111,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df_old = pd.read_csv(\"GPU_benchmarks_v7.csv\").reset_index(drop=True)\n",
        "df_new = pd.read_csv(\"cleaned_gpu.csv\").reset_index(drop=True)"
      ],
      "metadata": {
        "id": "BkowKNvBsgMH"
      },
      "execution_count": 112,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df_old.columns = df_old.columns.str.strip()\n",
        "df_new.columns = df_new.columns.str.strip()\n",
        "\n",
        "df_old.rename(columns={\"gpuName\": \"GPU_Model\",\n",
        "                       \"price\": \"Price\"}, inplace=True)\n",
        "\n",
        "df_new.rename(columns={\"Product\": \"GPU_Model\"}, inplace=True)"
      ],
      "metadata": {
        "id": "1UY2dfk0sm2Y"
      },
      "execution_count": 113,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def clean_price(x):\n",
        "    if pd.isna(x):\n",
        "        return np.nan\n",
        "    x = re.sub(r\"[^\\d.]\", \"\", str(x))\n",
        "    return float(x) if x != \"\" else np.nan\n",
        "\n",
        "df_old[\"Price\"] = df_old[\"Price\"].apply(clean_price)\n",
        "df_new[\"Price\"] = df_new[\"Price\"].apply(clean_price)"
      ],
      "metadata": {
        "id": "AssYgiQIsq1m"
      },
      "execution_count": 114,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df_old[\"Price\"]"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 458
        },
        "id": "wwaiWfRnsvVu",
        "outputId": "c171f727-0e16-4139-834c-d2a2552b00c7"
      },
      "execution_count": 116,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0       2099.99\n",
              "1       1199.99\n",
              "2       1749.99\n",
              "3       1120.31\n",
              "4        999.00\n",
              "         ...   \n",
              "2312        NaN\n",
              "2313        NaN\n",
              "2314        NaN\n",
              "2315        NaN\n",
              "2316        NaN\n",
              "Name: Price, Length: 2317, dtype: float64"
            ],
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Price</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>2099.99</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>1199.99</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>1749.99</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>1120.31</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>999.00</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>...</th>\n",
              "      <td>...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2312</th>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2313</th>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2314</th>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2315</th>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2316</th>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>2317 rows × 1 columns</p>\n",
              "</div><br><label><b>dtype:</b> float64</label>"
            ]
          },
          "metadata": {},
          "execution_count": 116
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df_new[\"Price\"]"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 458
        },
        "id": "-Id5yKa3s_w-",
        "outputId": "2f120ffa-ce54-46c2-892a-5249b88c3c54"
      },
      "execution_count": 117,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0      11200.0\n",
              "1       4890.0\n",
              "2      11800.0\n",
              "3       9150.0\n",
              "4      10600.0\n",
              "        ...   \n",
              "95    113000.0\n",
              "96     19990.0\n",
              "97     14600.0\n",
              "98      8990.0\n",
              "99     59900.0\n",
              "Name: Price, Length: 100, dtype: float64"
            ],
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Price</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>11200.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>4890.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>11800.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>9150.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>10600.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>...</th>\n",
              "      <td>...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>95</th>\n",
              "      <td>113000.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>96</th>\n",
              "      <td>19990.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>97</th>\n",
              "      <td>14600.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>98</th>\n",
              "      <td>8990.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>99</th>\n",
              "      <td>59900.0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>100 rows × 1 columns</p>\n",
              "</div><br><label><b>dtype:</b> float64</label>"
            ]
          },
          "metadata": {},
          "execution_count": 117
        }
      ]
    }
  ]
}