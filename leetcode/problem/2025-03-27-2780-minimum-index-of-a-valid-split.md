---
layout: leetcode-entry
title: "2780. Minimum Index of a Valid Split"
permalink: "/leetcode/problem/2025-03-27-2780-minimum-index-of-a-valid-split/"
leetcode_ui: true
entry_slug: "2025-03-27-2780-minimum-index-of-a-valid-split"
---

[2780. Minimum Index of a Valid Split](https://leetcode.com/problems/minimum-index-of-a-valid-split/description/) medium
[blog post](https://leetcode.com/problems/minimum-index-of-a-valid-split/solutions/6585144/kotlin-rust-by-samoylenkodmitry-j4aa/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/27032025-2780-minimum-index-of-a?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/qByXQ_HgU-U)
![1.webp](/assets/leetcode_daily_images/c2d20249.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/940

#### Problem TLDR

Min split index to have equal majority elements #medium #counting

#### Intuition

My intuition was to brute-force simulate the solution: count frequences, count frequencies `of frequencies`, put them in a sorted set:

```j

    // 2,1,3,1,1,1,7,1,2,1 f: 2->2,1->6,3->1,7->1; ff: 1->2,2->2,6->2
    // *                      2->1                     1->3,2->1
    //   *                         1->5                6->0,5->1
    //     *                            3->0           1->2
    //       *                     1->4                5->0,4->1
    //         *                   1->3                4->0,3->1

```
It was an ugly big solution, with time complexity of O(nlog(n)), but it was accepted.

Now, the helpful ideas:
* we only care about the current element frequency (not the others)
* there can only be a one such element, so precompute it
* we can find the majority element without a hashmap: a single counter that will reset element when it is `0`; the idea is derived from an observation, `the longest continuous subarray will be from the majority element, or example - ab ac ad ff, set majority element _before_ changing the counter`.

#### Approach

* two passes possible, one pass not so sure

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$ or O(1)

#### Code

```kotlin

    fun minimumIndex(n: List<Int>): Int {
        val f1 = n.groupBy { it }; val f2 = HashMap<Int, Int>()
        return n.withIndex().firstOrNull { (i, x) ->
            f2[x] = 1 + (f2[x] ?: 0); val f = f1[x]!!.size - f2[x]!!
            f2[x]!! * 2 > i + 1 && f * 2 > n.size - i - 1
        }?.index ?: -1
    }

```
```rust

    pub fn minimum_index(n: Vec<i32>) -> i32 {
        let (mut f, mut mi, mut mx, mut cm) = (0, -1, 0, 0);
        for &x in &n { if f == 0 { mx = x }; if x == mx { f += 1 } else { f -= 1 }}
        for i in 0..n.len() { if n[i] == mx { cm += 1;
            if mi < 0 && cm * 2 > i + 1 { mi = i as i32; f = cm }
        }} return if (cm - f) * 2 > n.len() - mi as usize - 1 { mi as _ } else { -1 }
    }

```
```c++

    int minimumIndex(vector<int>& n) {
        int f = 0, mi = -1, mx = 0, cm = 0;
        for (int x: n) { if (!f) mx = x; f += 1 - 2 * (x == mx); }
        for (int i = 0; i < size(n); ++i) if (n[i] == mx && ++cm * 2 > i + 1 && mi < 0) mi = i, f = cm;
        return (cm - f) * 2 > size(n) - mi - 1 ? mi : -1;
    }

```

