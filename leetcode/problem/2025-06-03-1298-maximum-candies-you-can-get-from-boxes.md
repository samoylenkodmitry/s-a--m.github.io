---
layout: leetcode-entry
title: "1298. Maximum Candies You Can Get from Boxes"
permalink: "/leetcode/problem/2025-06-03-1298-maximum-candies-you-can-get-from-boxes/"
leetcode_ui: true
entry_slug: "2025-06-03-1298-maximum-candies-you-can-get-from-boxes"
---

[1298. Maximum Candies You Can Get from Boxes](https://leetcode.com/problems/maximum-candies-you-can-get-from-boxes/description) hard
[blog post](https://leetcode.com/problems/maximum-candies-you-can-get-from-boxes/solutions/6806764/kotlin-rust-by-samoylenkodmitry-1ltr/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/03062025-1298-maximum-candies-you?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/S_UU1ZmxbDg)
![1.webp](/assets/leetcode_daily_images/10d7b0d8.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1008

#### Problem TLDR

Open boxes graph with keys simulation #hard

#### Intuition

Just the simulation steps in a BFS

#### Approach

* make sure `keys` didn't add unvisited box

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

// 5ms https://leetcode.com/problems/maximum-candies-you-can-get-from-boxes/submissions/1652437683
    fun maxCandies(st: IntArray, cs: IntArray, ks: Array<IntArray>, cb: Array<IntArray>, ib: IntArray): Int {
        val q = LinkedList<Int>(); var res = 0; for (b in ib) if (st[b] > 0) { st[b] = -1; q += b } else st[b] = 2
        while (q.size > 0) {
            val b = q.removeFirst(); res += cs[b]
            for (c in cb[b]) if (st[c] > 0) { st[b] = -1; q += c } else st[c] = 2
            for (k in ks[b]) if (st[k] > 1) { st[k] = -1; q += k } else if (st[k] == 0) st[k] = 1
        }
        return res
    }

```
```rust

// 0ms
    pub fn max_candies(mut st: Vec<i32>, cs: Vec<i32>, ks: Vec<Vec<i32>>, cb: Vec<Vec<i32>>, ib: Vec<i32>) -> i32 {
        let (mut q, mut r) = (VecDeque::from_iter(ib), 0);
        while let Some(b) = q.pop_front() { let b = b as usize;
            if st[b] > 0 {
                st[b] = -1; r += cs[b]; q.extend(&cb[b]);
                for &k in &ks[b] { let k = k as usize; if st[k] > 1 { q.push_back(k as i32) } else if st[k] == 0 { st[k] = 1 }}
            } else if st[b] == 0 { st[b] = 2 }
        } r
    }

```
```c++

// 0ms
    int maxCandies(vector<int>& st, vector<int>& cs, vector<vector<int>>& ks, vector<vector<int>>& cb, vector<int>& ib) {
        queue<int> q; int r = 0;
        for (int b: ib) if (st[b]) { st[b] = -1; q.push(b); } else st[b] = 2;
        while (size(q)) {
            int b = q.front(); q.pop(); r += cs[b];
            for (int c: cb[b]) if (st[c] > 0) { st[c] = -1; q.push(c); } else st[c] = 2;
            for (int k: ks[b]) if (st[k] > 1) { st[k] = -1; q.push(k); } else if (!st[k]) st[k] = 1;
        } return r;
    }

```

