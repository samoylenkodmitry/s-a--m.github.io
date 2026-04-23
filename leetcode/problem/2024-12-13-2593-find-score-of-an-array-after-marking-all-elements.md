---
layout: leetcode-entry
title: "2593. Find Score of an Array After Marking All Elements"
permalink: "/leetcode/problem/2024-12-13-2593-find-score-of-an-array-after-marking-all-elements/"
leetcode_ui: true
entry_slug: "2024-12-13-2593-find-score-of-an-array-after-marking-all-elements"
---

[2593. Find Score of an Array After Marking All Elements](https://leetcode.com/problems/find-score-of-an-array-after-marking-all-elements/description/) medium
[blog post](https://leetcode.com/problems/find-score-of-an-array-after-marking-all-elements/solutions/6142482/kotlin-rust-by-samoylenkodmitry-9cn5/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/13122024-2593-find-score-of-an-array?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/XIy6EnsikDs)
[deep-dive](https://notebooklm.google.com/notebook/d8e1fb77-6cfb-461a-b8e6-d7d0899ac58b/audio)
![1.webp](/assets/leetcode_daily_images/a00aaebc.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/831

#### Problem TLDR

Sum of minimums in order excluding siblings #medium #monotonic_stack

#### Intuition

The straightforward way is to sort and take one-by-one, marking taken elements.

The more interesting approach: for each decreasing sequence, we will take every 2nd starting from the smallest.

We can do this with a Stack, or even more simply with alterating sums.

#### Approach

* let's try to implement all approaches
* if you look at the code and it looks simple, know it was paid off with pain

#### Complexity

- Time complexity:
$$O(nlog(n))$$ or O(n)

- Space complexity:
$$O(n)$$ or O(1)

#### Code

```kotlin

    fun findScore(nums: IntArray): Long {
        var res = 0L; var s = Stack<Int>()
        for (n in nums + Int.MAX_VALUE)
            if (s.size > 0 && s.peek() <= n)
                while (s.size > 0) {
                    res += s.pop()
                    if (s.size > 0) s.pop()
                }
            else s += n
        return res
    }

```

```rust

    pub fn find_score(nums: Vec<i32>) -> i64 {
        let (mut r, mut a, mut b, mut l) = (0, 0, 0, i64::MAX);
        for n in nums {
            let n = n as i64;
            if l <= n {
                r += b; a = 0; b = 0; l = i64::MAX
            } else {
                (a, b) = (b, a + n); l = n
            }
        }; r + b
    }

```

```c++

    long long findScore(vector<int>& n) {
        long long r = 0; int e = n.size() - 1;
        vector<int> idx(n.size());
        iota(begin(idx), end(idx), 0);
        stable_sort(begin(idx), end(idx), [&](int i, int j) { return n[i] < n[j];});
        for (int i: idx) if (n[i])
            r += n[i], n[i] = n[min(e, i + 1)] = n[max(0, i - 1)] = 0;
        return r;
    }

```

