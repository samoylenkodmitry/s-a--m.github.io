---
layout: leetcode-entry
title: "1161. Maximum Level Sum of a Binary Tree"
permalink: "/leetcode/problem/2026-01-06-1161-maximum-level-sum-of-a-binary-tree/"
leetcode_ui: true
entry_slug: "2026-01-06-1161-maximum-level-sum-of-a-binary-tree"
---

[1161. Maximum Level Sum of a Binary Tree](https://leetcode.com/problems/maximum-level-sum-of-a-binary-tree/description) medium
[blog post](https://leetcode.com/problems/maximum-level-sum-of-a-binary-tree/solutions/7471349/kotlin-rust-by-samoylenkodmitry-mm59/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/06012026-1161-maximum-level-sum-of?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/qN-zWqqJuJ4)

![142eb616-0647-486f-ae28-ff010de3fe01 (1).webp](/assets/leetcode_daily_images/59e9c2fe.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1229

#### Problem TLDR

Min level max sum in tree #medium

#### Intuition

Use BFS or DFS.

#### Approach

* compute max only after all number are added in this level

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
// 34ms
    fun maxLevelSum(r: TreeNode?) = ArrayDeque<TreeNode>().let { q ->
        q += r!!; (1..25).maxBy { if (q.size < 1) -100001 else
            (1..q.size).sumOf { q.removeFirst().run {
                left?.let { q += it }; right?.let { q += it }; `val`
            }}
        }
    }
```
```rust
// 6ms
    pub fn max_level_sum(r: Option<Rc<RefCell<TreeNode>>>) -> i32 {
        let (mut m, mut ml) = ([0;27],0);
        fn dfs(m: &mut[i32], ml: &mut usize, l: usize, n: &Option<Rc<RefCell<TreeNode>>>) {
            let Some(n) = n else { return }; if l > 25 { return }; let n = n.borrow();
            m[l] += n.val; *ml = l.max(*ml); dfs(m, ml, l+1, &n.left); dfs(m, ml, l+1, &n.right)
        }
        dfs(&mut m, &mut ml, 0, &r); -(0..=ml).map(|i|(m[i],-(i as i32)-1)).max().unwrap().1
    }
```

