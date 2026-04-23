---
layout: leetcode-entry
title: "2558. Take Gifts From the Richest Pile"
permalink: "/leetcode/problem/2024-12-12-2558-take-gifts-from-the-richest-pile/"
leetcode_ui: true
entry_slug: "2024-12-12-2558-take-gifts-from-the-richest-pile"
---

[2558. Take Gifts From the Richest Pile](https://leetcode.com/problems/take-gifts-from-the-richest-pile/description/) medium
[blog post](https://leetcode.com/problems/take-gifts-from-the-richest-pile/solutions/6138685/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/12122024-2558-take-gifts-from-the?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/z7TkbUA51jU)
[deep-dive](https://notebooklm.google.com/notebook/639f9ca2-6b9f-49e8-a7c3-524d4d3e5d1e/audio)
![1.webp](/assets/leetcode_daily_images/5713b33e.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/830

#### Problem TLDR

Sum after `k-Sqrt` of tops in array #easy

#### Intuition

We can use a heap.

#### Approach

* some extra attention should be paid to use an sqrt: in Kotiln & Rust convert to Double, in Rust we aren't able to sort Doubles, so convert back.
* c++ is much more forgiving

#### Complexity

- Time complexity:
$$O(nlog(n))$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun pickGifts(gifts: IntArray, k: Int): Long {
        val pq = PriorityQueue(gifts.map { -it.toDouble() })
        for (i in 1..k) pq += -floor(sqrt(-pq.poll()))
        return -pq.sum().toLong()
    }

```

```rust

    pub fn pick_gifts(gifts: Vec<i32>, k: i32) -> i64 {
        let mut bh = BinaryHeap::from_iter(gifts);
        for i in 0..k {
            let x = bh.pop().unwrap();
            bh.push(((x as f64).sqrt()).floor() as i32)
        }
        bh.iter().map(|&x| x as i64).sum()
    }

```

```c++

    long long pickGifts(vector<int>& gifts, int k) {
        priority_queue<int> pq; long long res = 0;
        for (int g: gifts) pq.push(g);
        while (k--) {
            int x = pq.top(); pq.pop();
            pq.push(sqrt(x));
        }
        while (pq.size()) res += pq.top(), pq.pop();
        return res;
    }

```

