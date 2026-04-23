---
layout: leetcode-entry
title: "2551. Put Marbles in Bags"
permalink: "/leetcode/problem/2025-03-31-2551-put-marbles-in-bags/"
leetcode_ui: true
entry_slug: "2025-03-31-2551-put-marbles-in-bags"
---

[2551. Put Marbles in Bags](https://leetcode.com/problems/put-marbles-in-bags/description/) hard
[blog post](https://leetcode.com/problems/put-marbles-in-bags/solutions/6599042/kotlin-rust-by-samoylenkodmitry-ac4x/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/31032025-2551-put-marbles-in-bags?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/B-OnwniknU0)
![1.webp](/assets/leetcode_daily_images/7c18f8c0.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/944

#### Problem TLDR

Max split sum - min split sum of borders #hard #sort

#### Intuition

Let's look at the problem arithmetics:

```j

    // 1 3 5 1      k = 3  best way to split?
    // a b c c    pairs are included
    // 1+3       4
    //   3+5     8
    //     5+1   6    first and last - irrelevant
    //                                 or not?
    //                                 can take single
    // 1+1       2
    //   3+3     6
    //     5+5   10
    //       1+1 2

    // start with 3+3           should start with max?
    //       compare with 3+5           5+5
    //       compare with 3+1

    // start with single marble? 1 .. 1
    // firt split: at max pair   1..3+5..1
    // secons split: max pair    1..3+5..5+1..1, sum max = 1+3+5+5+1+1=16
    //               min pair    1..1+3..1
    //               min pair2   1..1+3..5+1.1 sum min=1+1+3+5+1+1=12
    //                         max-min=4

    // final sum?

```

* consider only pairs sums
* take them with priority
* sum them

#### Approach

* we can just sort them and take first k and last k
* or use a heap
* or quickselect (c++ solution), careful: can overlap

#### Complexity

- Time complexity:
$$O(nlogn)$$ or nlogk for heap or O(n) for quickselect (nk^2) worst

- Space complexity:
$$O(n)$$, can do in-place, O(1) with quickselect, or custom sort algorithm. O(logk) with default sort algorithm of k elements

#### Code

```kotlin

    fun putMarbles(w: IntArray, k: Int) = w.asList()
        .windowed(2).map { 1L * it.sum() }.sorted()
        .run { takeLast(k - 1).sum() - take(k - 1).sum() }

```
```kotlin

    fun putMarbles(w: IntArray, k: Int): Long {
        val pmax = PriorityQueue<Int>(); val pmin = PriorityQueue<Int>()
        return (0..w.size - 2).sumOf { i ->
            val s = w[i] + w[i + 1]; pmin += s; pmax += -s
            if (i > k - 2) -1L * pmin.poll() - pmax.poll() else 0
        }
    }

```
```rust

    pub fn put_marbles(w: Vec<i32>, k: i32) -> i64 {
        let mut w: Vec<_> = w.windows(2).map(|w| (w[0] + w[1]) as i64).collect();
        w.sort_unstable(); let k = k as usize - 1;
        w[w.len() - k..].iter().sum::<i64>() - w[..k].iter().sum::<i64>()
    }

```
```rust

    pub fn put_marbles(w: Vec<i32>, k: i32) -> i64 {
        let (mut pmax, mut pmin) = (BinaryHeap::new(), BinaryHeap::new());
        (0..w.len() - 1).map(|i| {
            let s = (w[i] + w[i + 1]) as i64; pmin.push(-s); pmax.push(s);
            if i + 2 > k as usize { pmax.pop().unwrap() + pmin.pop().unwrap() } else { 0 }
        }).sum::<i64>()
    }

```
```c++

    long long putMarbles(vector<int>& s, int k) {
        for (int i = 0; i < size(s) - 1; ++i) s[i] += s[i + 1]; s.pop_back();
        nth_element(s.begin(), s.begin() + (k - 1), s.end());
        long long sum_small = accumulate(s.begin(), s.begin() + (k - 1), 0LL);
        nth_element(s.begin(), s.end() - (k - 1), s.end());
        return accumulate(s.end() - (k - 1), s.end(), 0LL) - sum_small;
    }

```

