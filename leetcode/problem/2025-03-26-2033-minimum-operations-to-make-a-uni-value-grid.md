---
layout: leetcode-entry
title: "2033. Minimum Operations to Make a Uni-Value Grid"
permalink: "/leetcode/problem/2025-03-26-2033-minimum-operations-to-make-a-uni-value-grid/"
leetcode_ui: true
entry_slug: "2025-03-26-2033-minimum-operations-to-make-a-uni-value-grid"
---

[2033. Minimum Operations to Make a Uni-Value Grid](https://leetcode.com/problems/minimum-operations-to-make-a-uni-value-grid/description/) medium
[blog post](https://leetcode.com/problems/minimum-operations-to-make-a-uni-value-grid/solutions/6581014/kotlin-rust-by-samoylenkodmitry-7mlh/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/26032025-2033-minimum-operations?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/VnhOMBP9Hak)
![1.webp](/assets/leetcode_daily_images/847bef93.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/939

#### Problem TLDR

Min sum of differences of x times #medium #math #quickselect

#### Intuition

Didn't solve without a hint: the base reminder should be the same for all.

My first intuition is to count the operations by divinding by `x`:

```j

    //       *
    //       *
    // .-.-*-*
    // . * * *
    // * * * *
    // 1 2 3 5
    // 0 1 2 4   7
    // 1 0 1 3   5
    // 2 1 0 2   5
    // 3 2 1 1   7
    // 4 3 2 0   9

    // 0 0 0 0 0 10   10
    // 1 1 1 1 1 9    +5 -1 = 14
    // 2 2 2 2 2 8    +5 -1 = 18
    // 3 3 3 3 3 7    +5 -1 = 22
    // 4 4 4 4 4 6    +4 = 26
    // 5 5 5 5 5 5    +4 = 30
    // 6 6 6 6 6 4    +4 = 34

    // math problem?
    // binary search?
    // can be any base, hint 1, but it must be same for all

```
Then, there are some corner cases, where the number is not exactly divided by `x` but still have a valid solution. That's where the first hint applied.

The next hint is more about a math, the median is always the optimal distanced from all the numbers.

Or, we can search it manually: count numbers before and after the current. Move by one by one, and left sum will increase by the numbers before, and the right sum by the numbers after.

#### Approach

* instead of sorting, there is a quickselect algorithm

#### Complexity

- Time complexity:
$$O(nlog(n))$$, O(n) for quickselect, or line sweep

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun minOperations(g: Array<IntArray>, x: Int): Int {
        val q = ArrayList<Int>();
        for (r in g) for (n in r) { q += n / x; if (n % x != g[0][0] % x) return -1 }
        q.sort(); return q.sumOf { abs(it - q[q.size / 2]) }
    }

```
```rust

    pub fn min_operations(g: Vec<Vec<i32>>, x: i32) -> i32 {
        let (mut m, mut sa, mut sb, mut ca, mut cb, mut res) =
            ([0; 10_002], 0, 0, 0, 0, 0);
        for r in &g { for &n in r { if n % x != g[0][0] % x { return -1 };
            m[1 + (n / x) as usize] += 1; ca += 1; sa += 1 + n / x; res += n / x }}
        for i in 0..10_001 {
            sa -= ca; ca -= m[i + 1]; cb += m[i]; sb += cb; res = res.min(sa + sb);
        } res
    }

```
```c++

    int minOperations(vector<vector<int>>& g, int x) {
        vector<int> q;
        for (auto& r: g) for (int n: r) {
            if (n % x != g[0][0] % x) return -1;
            q.push_back(n / x);
        }
        nth_element(begin(q), begin(q) + size(q) / 2, end(q));
        int m = q[size(q) / 2], res = 0;
        for (int n: q) res += abs(n - m); return res;
    }

```

