#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <time.h>

// 個数の上限はあらかじめ定めておく
const int max_items = 1000;

const int rand_seed = 10;

// 以下は構造体の定義と関数のプロトタイプ宣言

// 構造体 itemset
// number分の価値value と 重さweight を格納しておく
// データをポインタで定義しており、mallocする必要あり
typedef struct itemset
{
  int number;
  double *value;
  double *weight;
} Itemset;

typedef struct wolve
{
  int *flags;
  double *position;
  int cluster_id;
  double fitness;
} Wolve;

typedef struct cluster
{
  int cluser_id;
  double *centroid;
  Wolve *x_alpha;
  Wolve *x_beta;
  Wolve *x_gamma;
} Cluster;

typedef struct wolves
{
  Wolve *wolves;
} Wolves;

Wolve *best_address;

// 関数のプロトサイプ宣言

// Itemset *init_itemset(int, int);
//
// itemsetを初期化し、そのポインタを返す関数
// 引数:
//  品物の個数: number (int)
//  乱数シード: seed (int) // 品物の値をランダムにする
// 返り値:
//  確保されたItemset へのポインタ
Itemset *init_itemset(int number);

// void free_itemset();

// Itemset *load_itemset(char *filename)
//
// ファイルからItemset を設定し、確保された領域へのポインタを返す関数 [未実装]
// 引数:
//  Itemsetの必要パラメータが記述されたバイナリファイルのファイル名 filename (char*)
// 返り値:
//  Itemset へのポインタ
Itemset *load_itemset(char *filename);

// void print_itemset(const Itemset *list)
//
// Itemsetの内容を標準出力に表示する関数
void print_itemset(const Itemset *list);

// void save_itemset(char *filename)
//
// Itemsetのパラメータを記録したバイナリファイルを出力する関数
// 引数:
// Itemsetの必要パラメータを吐き出すファイルの名前 filename (char*)
// 返り値:
//  なし
void save_itemset(char *filename);

double rand_01()
{
  return rand() / (RAND_MAX * 1.0);
}

double sigmoid(double x)
{
  return 1.0 / (1.0 + exp((-1.0) * x));
}

int position_to_discrete(double position)
{
  if (position < 0.5)
  {
    return 0;
  }
  else
  {
    return 1;
  }
}
void calculate_fitness(Wolve *wolve, int dimension, Itemset *items, double W)
{
  double total_value = 0;
  double total_weight = 0;
  for (int i = 0; i < dimension; i++)
  {
    total_value += wolve->flags[i] * items->value[i];
    total_weight += wolve->flags[i] * items->weight[i];
  }
  if (total_weight > W)
  {
    total_value = 0;
  }
  wolve->fitness = total_value;
}

Wolves *initialize_wolves(int wolve_size, int dimension, Itemset *items, double W)
{
  Wolves *wolves;
  wolves = (Wolves *)malloc(sizeof(Wolves));
  wolves->wolves = (Wolve *)malloc(sizeof(Wolve) * wolve_size);
  for (int i = 0; i < wolve_size; i++)
  {
    wolves->wolves[i].position = (double *)malloc(sizeof(double) * dimension);
    wolves->wolves[i].flags = (int *)malloc(sizeof(int) * dimension);
    for (int j = 0; j < dimension; j++)
    {
      wolves->wolves[i].position[j] = rand_01();
      wolves->wolves[i].flags[j] = position_to_discrete(wolves->wolves[i].position[j]);
      calculate_fitness(&(wolves->wolves[i]), dimension, items, W);
    }
  }
  return wolves;
}

void calculate_allover_fitness(Wolves *wolves, int wolve_size, int dimension, Itemset *items, double W)
{
  for (int i = 0; i < wolve_size; i++)
  {
    for (int j = 0; j < dimension; j++)
    {
      wolves->wolves[i].flags[j] = position_to_discrete(wolves->wolves[i].position[j]);
      calculate_fitness(&(wolves->wolves[i]), dimension, items, W);
    }
  }
}

Cluster *initialize_cluster(Wolves *wolves, int wolve_size, int cluster_size, int dimension)
{
  Cluster *clusters;
  clusters = (Cluster *)malloc(sizeof(Cluster) * cluster_size);
  for (int i = 0; i < cluster_size; i++)
  {
    clusters[i].cluser_id = i;
    clusters[i].centroid = (double *)malloc(sizeof(double) * dimension);
  }
  for (int i = 0; i < wolve_size; i++)
  {
    int random = (int)floor(rand_01() * cluster_size);
    wolves->wolves[i].cluster_id = random;
  }
  return clusters;
}

void K_means(Wolves *wolves, Cluster *clusters, int wolve_size, int dimension, int cluster_size)
{
  Cluster *tmp_clusters = (Cluster *)malloc(sizeof(Cluster) * cluster_size);
  int *cluster_content_count = (int *)calloc(cluster_size, sizeof(int));

  for (int i = 0; i < cluster_size; i++)
  {
    tmp_clusters[i].centroid = (double *)malloc(sizeof(double) * dimension);
  }
  for (int i = 0; i < wolve_size; i++)
  {
    for (int j = 0; j < dimension; j++)
    {
      tmp_clusters[wolves->wolves[i].cluster_id].centroid[j] += wolves->wolves[i].position[j];
    }
    cluster_content_count[wolves->wolves[i].cluster_id]++;
  }

  for (int i = 0; i < cluster_size; i++)
  {
    for (int j = 0; j < dimension; j++)
    {
      tmp_clusters[i].centroid[j] = tmp_clusters[i].centroid[j] / (cluster_content_count[i] * 1.0);
      clusters[i].centroid[j] = tmp_clusters[i].centroid[j];
    }
  }

  for (int i = 0; i < wolve_size; i++)
  {
    double shortest_norm = RAND_MAX;
    for (int j = 0; j < cluster_size; j++)
    {
      double tmp_norm = 0;
      for (int k = 0; k < dimension; k++)
      {
        tmp_norm += pow(clusters[j].centroid[k] - wolves->wolves[i].position[k], 2);
      }
      if (tmp_norm < shortest_norm)
      {
        shortest_norm = tmp_norm;
        wolves->wolves[i].cluster_id = j;
      }
    }
  }
  free(tmp_clusters);
  free(cluster_content_count);
}

void item_move(Wolve **alpha_wolves, Wolve **beta_wolves, Wolve **gamma_wolves, int move_index, int cluster_index)
{
  if (move_index == 1)
  {
    beta_wolves[cluster_index] = alpha_wolves[cluster_index];
  }
  else if (move_index == 2)
  {
    gamma_wolves[cluster_index] = beta_wolves[cluster_index];
  }
}

int recursion(Wolve **X_wolves, Wolves *wolves, int wolve_index, int cluster_index, int X_index, Wolve **alpha_wolves, Wolve **beta_wolves, Wolve **gamma_wolves)
{
  if (X_wolves[cluster_index] == NULL)
  {
    X_wolves[cluster_index] = &(wolves->wolves[wolve_index]);
    return 1;
  }
  else
  {
    if (X_wolves[cluster_index]->fitness < wolves->wolves[wolve_index].fitness)
    {
      if (X_index == 1)
      {
        item_move(alpha_wolves, beta_wolves, gamma_wolves, 2, cluster_index);
        item_move(alpha_wolves, beta_wolves, gamma_wolves, 1, cluster_index);
      }
      else if (X_index == 2)
      {
        item_move(alpha_wolves, beta_wolves, gamma_wolves, 2, cluster_index);
      }
      X_wolves[cluster_index] = &(wolves->wolves[wolve_index]); //一個下のwolveが移されてない
      return 1;
    }
    else
    {
      return 0;
    }
  }
}

void find_best(Wolves *wolves, Cluster *clusters, int wolve_size, int dimension, int cluster_size)
{
  Wolve **alpha_wolves = (Wolve **)malloc(sizeof(Wolve *) * cluster_size);
  Wolve **beta_wolves = (Wolve **)malloc(sizeof(Wolve *) * cluster_size);
  Wolve **gamma_wolves = (Wolve **)malloc(sizeof(Wolve *) * cluster_size);

  for (int i = 0; i < cluster_size; i++)
  {
    alpha_wolves[i] = NULL;
    beta_wolves[i] = NULL;
    gamma_wolves[i] = NULL;
  }

  for (int i = 0; i < wolve_size; i++)
  {
    if (recursion(alpha_wolves, wolves, i, wolves->wolves[i].cluster_id, 1, alpha_wolves, beta_wolves, gamma_wolves) == 0)
    {
      if (recursion(beta_wolves, wolves, i, wolves->wolves[i].cluster_id, 2, alpha_wolves, beta_wolves, gamma_wolves) == 0)
      {
        recursion(gamma_wolves, wolves, i, wolves->wolves[i].cluster_id, 3, alpha_wolves, beta_wolves, gamma_wolves);
      }
    }
  }

  for (int i = 0; i < cluster_size; i++)
  {
    clusters[i].x_alpha = alpha_wolves[i];
    clusters[i].x_beta = beta_wolves[i];
    clusters[i].x_gamma = gamma_wolves[i];
  }
  free(alpha_wolves);
  free(beta_wolves);
  free(gamma_wolves);
}

void update_aAC(double *a, double *A, double *C, int iteration_index, int iteration_time, int dimension)
{
  *a = 2 - 2 * iteration_index / (iteration_time * 1.0);
  for (int i = 0; i < dimension; i++)
  {
    A[i] = 2 * (*a) * rand_01() - (*a);
    C[i] = 2 * rand_01();
  }
}

void search_one_wolve(Wolve *wolve, Cluster *clusters, int dimension, double *A, double *C, double *a, int iteration_index, int iteration_time)
{
  double *D_alpha = (double *)malloc(sizeof(double) * dimension);
  double *D_beta = (double *)malloc(sizeof(double) * dimension);
  double *D_gamma = (double *)malloc(sizeof(double) * dimension);
  double *X1 = (double *)malloc(sizeof(double) * dimension);
  double *X2 = (double *)malloc(sizeof(double) * dimension);
  double *X3 = (double *)malloc(sizeof(double) * dimension);

  update_aAC(a, A, C, iteration_index, iteration_time, dimension);
  for (int i = 0; i < dimension; i++)
  {
    D_alpha[i] = fabs(C[i] * (clusters[wolve->cluster_id].x_alpha)->position[i] - wolve->position[i]);
    X1[i] = (clusters[wolve->cluster_id].x_alpha)->position[i] - A[i] * D_alpha[i];
  }
  update_aAC(a, A, C, iteration_index, iteration_time, dimension);
  for (int i = 0; i < dimension; i++)
  {
    D_beta[i] = fabs(C[i] * (clusters[wolve->cluster_id].x_beta)->position[i] - wolve->position[i]);
    X2[i] = (clusters[wolve->cluster_id].x_beta)->position[i] - A[i] * D_beta[i];
  }
  update_aAC(a, A, C, iteration_index, iteration_time, dimension);
  for (int i = 0; i < dimension; i++)
  {
    D_gamma[i] = fabs(C[i] * (clusters[wolve->cluster_id].x_gamma)->position[i] - wolve->position[i]);
    X3[i] = (clusters[wolve->cluster_id].x_gamma)->position[i] - A[i] * D_gamma[i];
  }

  for (int i = 0; i < dimension; i++)
  {
    double cent_grav = (X1[i] + X2[i] + X3[i]) / 3.0;
    if (cent_grav > 1)
    {
      cent_grav = 1;
    }
    else if (cent_grav < 0)
    {
      cent_grav = 0;
    }
    wolve->position[i] = cent_grav;
  }
  free(D_alpha);
  free(D_beta);
  free(D_gamma);
  free(X1);
  free(X2);
  free(X3);
}

void search(Wolves *wolves, Cluster *clusters, int wolve_size, int dimension, double *A, double *C, double *a, int iteration_index, int iteration_time)
{
  for (int i = 0; i < wolve_size; i++)
  {
    if (clusters[wolves->wolves[i].cluster_id].x_alpha == best_address)
    {
      //printf("OK");
    }
    if ((clusters[wolves->wolves[i].cluster_id].x_alpha != &(wolves->wolves[i])) && (clusters[wolves->wolves[i].cluster_id].x_beta != &(wolves->wolves[i])) && (clusters[wolves->wolves[i].cluster_id].x_gamma != &(wolves->wolves[i])))
    {
      search_one_wolve(&(wolves->wolves[i]), clusters, dimension, A, C, a, iteration_index, iteration_time);
    }
    else
    {
      //printf("f");
    }
  }
  //printf("\n");
}

int print_status(Wolves *wolves, Cluster *clusters, int wolve_size, int dimension, int cluster_size)
{
  /*int counter = 0;
  for (int i = 0; i < wolve_size; i++)
  {
    for (int j = 0; j < dimension; j++)
    {
      printf("%d", wolves->wolves[i].flags[j]);
    }
    printf("  ");
    for (int j = 0; j < dimension; j++)
    {
      printf("%lf", wolves->wolves[i].position[j]);
    }
    printf(" %lf\n", wolves->wolves[i].fitness);

    printf(" cluster id=%d\n", wolves->wolves[i].cluster_id);
    if (wolves->wolves[i].cluster_id == 1)
    {
      counter++;
    }
  }
  printf("counter=%d\n", counter);*/
  double best_fitness = 0;

  for (int i = 0; i < cluster_size; i++)
  {
    /*for (int j = 0; j < dimension; j++)
    {
      printf("%lf ", clusters[i].centroid[j]);
    }*/
    if (clusters[i].x_alpha != NULL && (clusters[i].x_alpha)->fitness > best_fitness)
    {
      best_fitness = (clusters[i].x_alpha)->fitness;
      best_address = clusters[i].x_alpha;
    }
    /*if (clusters[i].x_alpha != NULL)
    {
      printf(" %lf", (clusters[i].x_alpha)->fitness);
    }
    else
    {
      printf(" 0.00");
    }

    if (clusters[i].x_beta != NULL)
    {
      printf(" %lf", (clusters[i].x_beta)->fitness);
    }
    else
    {
      printf(" 0.00");
    }

    if (clusters[i].x_gamma != NULL)
    {
      printf(" %lf", (clusters[i].x_gamma)->fitness);
    }
    else
    {
      printf(" 0.00");
    }

    printf("\n");*/
  }
  printf("best = %lf\n", best_fitness);
  if (best_fitness == 13549094)
  {
    return 1;
  }
  else
  {
    return 0;
  }
}

int main(int argc, char **argv)
{
  clock_t start, end;

  /* 引数処理: ユーザ入力が正しくない場合は使い方を標準エラーに表示して終了 */
  if (argc != 3)
  {
    fprintf(stderr, "usage: %s <the number of items (int)> <max capacity (double)>\n", argv[0]);
    exit(1);
  }

  const int dimension = atoi(argv[1]);
  assert(dimension <= max_items);
  const double W = atof(argv[2]);
  srand(rand_seed);

  start = clock();

  Cluster *clusters;
  Wolves *wolves;

  const int wolve_size = dimension * 10;
  const int cluster_size = ceil(wolve_size / 12.0);
  printf("cluster_size=%d\n", cluster_size);
  double a;
  double *A = (double *)malloc(sizeof(double) * dimension);
  double *C = (double *)malloc(sizeof(double) * dimension);
  int iteration_time = 500; //bprder between 13 and 14 segmentation fault

  printf("max capacity: W = %.f, # of items: %d\n", W, dimension);

  Itemset *items = init_itemset(dimension);
  print_itemset(items);

  wolves = initialize_wolves(wolve_size, dimension, items, W);
  clusters = initialize_cluster(wolves, wolve_size, cluster_size, dimension);
  for (size_t i = 0; i < 3; i++)
  {
    K_means(wolves, clusters, wolve_size, dimension, cluster_size);
  }
  find_best(wolves, clusters, wolve_size, dimension, cluster_size);

  print_status(wolves, clusters, wolve_size, dimension, cluster_size);

  for (int i = 0; i < iteration_time; i++)
  {
    search(wolves, clusters, wolve_size, dimension, A, C, &a, i, iteration_time);
    calculate_allover_fitness(wolves, wolve_size, dimension, items, W);
    find_best(wolves, clusters, wolve_size, dimension, cluster_size);
    int flag = print_status(wolves, clusters, wolve_size, dimension, cluster_size);
    if (flag == 1)
    {
      break;
    }
  }

  end = clock();

  printf("%.2f秒かかりました\n", (double)(end - start) / CLOCKS_PER_SEC);

  return 0;
}

// 構造体をポインタで確保するお作法を確認してみよう
Itemset *init_itemset(int number)
{
  Itemset *list = (Itemset *)malloc(sizeof(Itemset));

  list->number = number;

  list->value = (double *)malloc(sizeof(double) * number);
  list->weight = (double *)malloc(sizeof(double) * number);

  FILE *fp_value;
  FILE *fp_weight;

  if ((fp_value = fopen("p08_p.txt", "r")) == NULL)
  {
    printf("p07_p.txt error");
    exit(1);
  }
  if ((fp_weight = fopen("p08_w.txt", "r")) == NULL)
  {
    printf("p07_w.txt error");
    exit(1);
  };

  for (int i = 0; i < number; i++)
  {
    double *tmp_v = (double *)malloc(sizeof(double));
    double *tmp_w = (double *)malloc(sizeof(double));
    fscanf(fp_value, "%lf", tmp_v);
    fscanf(fp_weight, "%lf", tmp_w);
    list->value[i] = *tmp_v;
    list->weight[i] = *tmp_w;
  }
  fclose(fp_value);
  fclose(fp_weight);

  return list;
}

// itemset の free関数
void free_itemset(Itemset *list)
{
  free(list->value);
  free(list->weight);
  free(list);
}

// 表示関数
void print_itemset(const Itemset *list)
{
  int n = list->number;
  for (int i = 0; i < n; i++)
  {
    printf("v[%d] = %4.1f, w[%d] = %4.1f\n", i, list->value[i], i, list->weight[i]);
  }
  printf("----\n");
}
