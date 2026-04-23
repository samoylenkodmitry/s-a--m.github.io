---
layout: leetcode-entry
title: "2561. Rearranging Fruits"
permalink: "/leetcode/problem/2025-08-02-2561-rearranging-fruits/"
leetcode_ui: true
entry_slug: "2025-08-02-2561-rearranging-fruits"
---

[2561. Rearranging Fruits](https://leetcode.com/problems/rearranging-fruits/description/) hard
[blog post](https://leetcode.com/problems/rearranging-fruits/solutions/7035793/kotlin-by-samoylenkodmitry-k7rr/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/2082025-2561-rearranging-fruits?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/f5ehPg4kNio)
![1.webp](/assets/leetcode_daily_images/77a9592e.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1068

#### Problem TLDR

Min swaps cost to make a==b #hard #greedy

#### Intuition

Didn't solved

```j
    // first idea:
    // count freq1, freq2
    // f1[i] % 2 != 0, f2[j] % 2 != 0
    // in f1 and not in f2 (split by half)
    // in f2 and not in f1 (split by half)
    // f1[x] != f2[x] then move
    // 1: 2            0    split, cost = 1 * 2/2 = 1
    // 2: 3            1    move, 3 - (3+1)/2, b[i] * (max(f1,f2) - (f1+f2)/2)
    // we have array of costs:, count should be % 2
    // c1 c2 c3 c4 and have to pick pairs (min,max),
    // sort, then two pointers
    // wrong answer (14 minute)
    // [84,80,43,8,80,88,43,14,100,88]
    // [32,32,42,68,68,100,42,84,14,8]
    // {84=1, 80=2, 43=2, 8=1, 88=2, 14=1, 100=1}
    // {32=2, 42=2, 68=2, 100=1, 84=1, 14=1, 8=1}
    // 8, 14, 43, 43, 80, 80, 84, 88, 88, 100   a
    // 8, 14, 32, 32, 42, 42, 68, 68, 84, 100   b
    // move x=80 a=2 b=0, cost=80
    // move x=43 a=2 b=0, cost=43
    // move x=88 a=2 b=0, cost=88
    // move x=32 a=0 b=2, cost=32
    // move x=42 a=0 b=2, cost=42
    // move x=68 a=0 b=2, cost=68
    // [32, 42, 43, 68, 80, 88]
    //  b   b   a   b   a   a
    // 43 80 88
    // 32 42 68    32,88 + 43,68 + 42,80 = 32+43+42 = 32+85 = 117
    // wrong answer how is it 48? where is 48 from?
    // took hints (29 minutes)
    // the hint in the comments: `use the minimum element 8 to do 6 swaps.` (what??)
    // how does indirect swap works?
    //
    // 2 2 100 100
    // 3 3 200 200
    //
    // 3 2 100 100       2
    // 3 2 200 200
    //
    // 2 200 100 100     2
    // 3 3 200 2
    //
    // 2 200 2 100       2
    // 3 3 200 100
    //
    // 2 200 3 100       2         2x4=8
    // 3 2 200 100
    //
    // or
    // 200 2 100 100     2
    // 3 3 2 200
    //
    // 200 2 3 100       3      2+3=5
    // 100 3 2 200

    // 1 100 100
    // 1 200 200
    //
    // 200 100 100     1
    // 1 1 200
    //
    // 200 1 100       1
    // 1 100 200
    //
    // 4 4 4 4 3
    //  3
    //
    // 4 4 4 3 3       3
    // 5 5 5 5 4
    //
    // 4 4 4 3 5       3
    // 5 5 5 3 4
    //
    // 4 4 4 5 5       3
    // 5 5 3 3 4
    //
    // 4 4 3 5 5       3
    // 5 5 3 4 4             4x3=12
    //
    // another corner case if smallest itself in the wrong position
    // 28 wrong positions, ans smallest here, so 27

```j

What went wrong:
* I was trying to cut corners with some complex `algorithm` to make greedy work with entire groups
* however, with greedy, we have to simulate each step individually, hence make a greedy choice for each swapped value, between itself and `2 * min` jump

#### Approach

* the solution from https://leetcode.com/problems/rearranging-fruits/solutions/3143735/ordered-map/ has mind blowing trick: `min(sw, abs(f) / 2)`; we `assume` that the optimal swaps count can't be more than `sw` (swaps from one side to another)
* to make greedy optimization for groups is to make at most `sw` swaps

#### Complexity

- Time complexity:
$$O(nlog(n))$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

// 101ms
    fun minCost(b1: IntArray, b2: IntArray): Long {
        val f1 = b1.groupBy { it }; val f2 = b2.groupBy { it }
        val min = min(b1.min(), b2.min())
        return (f1.keys + f2.keys).flatMap { x ->
            val a = f1[x]?.size ?: 0; val b = f2[x]?.size ?: 0;
            if ((a + b) % 2 > 0) return -1L
            List(abs(a - b) / 2) { x }
        }.run { sorted().take(size/2).sumOf { 1L * min(it, 2 * min) }}
    }

```
```kotlin

// 71ms
    fun minCost(a: IntArray, b: IntArray): Long {
        val m = TreeMap<Int, Int>(); var sw = 0; var r = 0L; val min = min(a.min(), b.min())
        for (x in a) m[x] = (m[x] ?: 0) + 1; for (x in b) m[x] = (m[x] ?: 0) - 1
        for ((x, f) in m) { if (f % 2 != 0) return -1L; sw += max(0, f / 2) }
        for ((x, f) in m) {
            val take = min(sw, abs(f) / 2)
            r += 1L * take * min(x, min * 2)
            sw -= take; if (sw == 0) break
        }
        return r
    }

```

