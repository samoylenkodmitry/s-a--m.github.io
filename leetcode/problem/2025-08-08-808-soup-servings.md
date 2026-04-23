---
layout: leetcode-entry
title: "808. Soup Servings"
permalink: "/leetcode/problem/2025-08-08-808-soup-servings/"
leetcode_ui: true
entry_slug: "2025-08-08-808-soup-servings"
---

[808. Soup Servings](https://leetcode.com/problems/soup-servings/description/) medium
[blog post](https://leetcode.com/problems/soup-servings/solutions/7057068/kotlin-rust-by-samoylenkodmitry-8a2i/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/8082025-808-soup-servings?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/teBK2YRCO3M)
![1.webp](/assets/leetcode_daily_images/15f5cc29.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1074

#### Problem TLDR

Probability A empty first plus both empty /2 #medium #dp #probability

#### Intuition

The probability is the number of good events divided by the total number of events.
Each layer deep result should be divided by current layer number of events: `p_curr = (p deep) / curr_total_events`
```j
    // 50 50
    // 1: 100+0 - end  A first
    // 2: 70+25 - end  A first
    // 3: 50+50 - end  both
    // 4: 25+75 - end  B first
    // P(a first) = 2/4
    // P(both) = 1/4
    // 2/4 + 1/2 * 1/4 = 1/4 * (2 + 0.5) = 0.625

    // 100 100
    // 1: 100+0 A first
    // 2: 70+25     30 75
    //              1: 100+0      A first
    //              2: 70+25      A first
    //              3: 50+50      A first
    //              4: 20+75      B first
    //              pa = 3/4 pboth = 0   p = 3/4
    // 3: 50+50     50  50
    //              1: 100+0      A first
    //              2: 70+25      A first
    //              3: 50+50      both
    //              4: 20+75      B first
    //              pa = 2/4 pboth = 1/4 p = 1/4 * (2+0.5)
    // 4: 25+75     75    25
    //              1: 100+0      A first
    //              2: 70+25      B first
    //              3: 50+50      B first
    //              4: 20+75      B first
    //              pa = 1/4 pboth=0    p=1/4
    //
    // p = 1/4 * (1 + 3/4 + 1/4*(2+0.5) + 1/4)
    //   = 1/4 * (2 + 1/4 + 2.5/4) = 0.71875
    // can't use n directly, too big 10^9 660295675
    // maybe after some number the answer is always 1

```

#### Approach

* just remember that on big numbers probabilities distribution collapses into the final value

#### Complexity

- Time complexity:
$$O(min(5000, n))$$,

- Space complexity:
$$O(min(5000, n))$$

#### Code

```kotlin

// 8ms
    val dp = HashMap<Pair<Int, Int>, Double>()
    fun soupServings(a: Int, b: Int = a): Double = if (a > 5000) 1.0
        else 0.25 * dp.getOrPut(a to b) {
            if (a <= 0 && b <= 0) 2.0 else if (a <= 0) 4.0 else if (b <= 0) 0.0
            else soupServings(a - 100, b) + soupServings(a - 75, b - 25) +
                 soupServings(a - 50, b - 50) + soupServings(a - 25, b - 75)
        }

```
```kotlin

// 5ms
    fun soupServings(n: Int): Double {
        if (n > 5000) return 1.0
        val N = 4 + (n+24) / 25; val p = Array(N) { DoubleArray(N) }
        for (j in 0..4) for (i in j..<N) { p[j][i] = 1.0; p[j][j] = 0.5 }
        for (a in 4..<N) for (b in 4..<N) p[a][b] = 0.25 *
        (p[a-4][b] + p[a-3][b-1] + p[a-2][b-2] + p[a-1][b-3])
        return p[N-1][N-1]
    }

```
```rust

// 0ms
    pub fn soup_servings(n: i32) -> f64 {
        if n > 5000 { return 1.0 }
        let n = ((n+124)/25) as usize; let mut p = vec![vec![0.0;n];n];
        for j in 0..5 { for i in j..n { p[j][i] = 1.0; p[j][j] = 0.5 }}
        for a in 4..n { for b in 4..n { p[a][b] = 0.25 *
        (p[a-4][b] + p[a-3][b-1] + p[a-2][b-2] + p[a-1][b-3])}}; p[n-1][n-1]
    }

```
```c++

// 0ms
    double soupServings(int n) {
        if (n > 5000) return 1.0;
        int m = (n + 124)/25; double p[201][201]={0};
        for (int j = 0; j < 5; ++j) for (int i = j; i < m; ++i) p[j][i]=1.0, p[j][j]=0.5;
        for (int a = 4; a < m; ++a) for (int b = 4; b < m; ++b) p[a][b] = 0.25 *
        (p[a-4][b] + p[a-3][b-1] + p[a-2][b-2] + p[a-1][b-3]);
        return p[m-1][m-1];
    }

```
```python

// 9ms
    def soupServings(self, n: int) -> float:
        if n > 5000: return 1.0
        m = (n + 124) // 25
        p = [[0.0]*(m + 1) for _ in range(m + 1)]
        for j in range(5):
            for i in range(j, m): p[j][i] = 1.0
            p[j][j] = 0.5
        for a in range(4, m):
            for b in range(4, m):
                p[a][b] = 0.25*(p[a-4][b] + p[a-3][b-1] + p[a-2][b-2] + p[a-1][b-3])
        return p[m-1][m-1]

```

