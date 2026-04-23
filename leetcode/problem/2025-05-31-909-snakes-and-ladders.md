---
layout: leetcode-entry
title: "909. Snakes and Ladders"
permalink: "/leetcode/problem/2025-05-31-909-snakes-and-ladders/"
leetcode_ui: true
entry_slug: "2025-05-31-909-snakes-and-ladders"
---

[909. Snakes and Ladders](https://leetcode.com/problems/snakes-and-ladders/description/) medium
[blog post](https://leetcode.com/problems/snakes-and-ladders/solutions/6798315/kotlin-rust-by-samoylenkodmitry-oe2h/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/31052025-909-snakes-and-ladders?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/t30eV_c9kw0)
![1.webp](/assets/leetcode_daily_images/3ce1ad50.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1005

#### Problem TLDR

Shortest path bottom-top in zig-zag matrix with jumps #medium #bfs

#### Intuition

Surprisingly, didn't covered all corener cases.

```j
    // 1:15, still some corner case not covered, looking for solutions....

```

My issue was a premature optimization, trying to inline visited set with jumps.
After making a separate visited set it all worked out.

#### Approach

* LinkedList vs ArrayDeque is 5ms vs 20ms drop-in replacement difference in Kotlin
* don't premature optimize on the first go
* read instructions slowly: we never do jump-jump, even in 2 ticks

#### Complexity

- Time complexity:
$$O(n^2)$$

- Space complexity:
$$O(n^2)$$

#### Code

```kotlin

// 6ms
    fun snakesAndLadders(b: Array<IntArray>): Int {
        var s = -1; val n = b.size; var (q, q1) = List(2) { ArrayList<Int>(400) }; q1 += 1
        while (q1.size > 0 && ++s >= 0.also { q = q1.also { q1 = q }; q1.clear() })
            for (i in q) for (j in i + 1..min(i + 6, n * n)) {
                val y = n - 1 - (j - 1) / n
                val x = if (y % 2 != n % 2) (j - 1) % n else n - 1 - (j - 1) % n
                if (b[y][x] < -1) continue
                val k = if (b[y][x] >= 0) b[y][x] else j; b[y][x] = -2; q1 += k
                if (k == n * n) return s + 1
            }
        return -1
    }

```
```rust

// 0ms
    pub fn snakes_and_ladders(mut b: Vec<Vec<i32>>) -> i32 {
        let (mut s, mut n, mut q, mut q1) = (0, b.len(), vec![1], vec![]);
        while q.len() > 0 {
            for &i in &q { for j in i + 1..=(i + 6).min(n * n) {
                let y = n - 1 - (j - 1) / n;
                let x = if y % 2 != n % 2 { (j - 1) % n } else { n - 1 - (j - 1) % n };
                if b[y][x] < -1 { continue }
                let k = if b[y][x] >= 0 { b[y][x] } else { j as i32 }; b[y][x] = -2;
                if k == (n * n) as i32 { return s + 1 }
                q1.push(k as usize) }}
            s += 1; (q, q1) = (q1, q); q1.clear();
        }; -1
    }

```
```c++

// 0ms
    int snakesAndLadders(vector<vector<int>>& b) {
        int n = b.size(), s = -1; vector<int> q, q1 = {1};
        while (!q1.empty()) {
            s++; q.swap(q1); q1.clear();
            for (int i : q) for (int j = i + 1, end = min(i + 6, n * n); j <= end; j++) {
                int y = n - 1 - (j - 1) / n;
                int x = (y % 2 != n % 2) ? (j - 1) % n : n - 1 - (j - 1) % n;
                if (b[y][x] < -1) continue;
                int k = (b[y][x] >= 0 ? b[y][x] : j); b[y][x] = -2;
                if (k == n * n) return s + 1; q1.push_back(k);
            }
        }
        return -1;
    }

```

