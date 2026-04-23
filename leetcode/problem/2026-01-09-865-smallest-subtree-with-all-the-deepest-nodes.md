---
layout: leetcode-entry
title: "865. Smallest Subtree with all the Deepest Nodes"
permalink: "/leetcode/problem/2026-01-09-865-smallest-subtree-with-all-the-deepest-nodes/"
leetcode_ui: true
entry_slug: "2026-01-09-865-smallest-subtree-with-all-the-deepest-nodes"
---

[865. Smallest Subtree with all the Deepest Nodes](https://leetcode.com/problems/smallest-subtree-with-all-the-deepest-nodes/description) medium
[blog post](https://leetcode.com/problems/smallest-subtree-with-all-the-deepest-nodes/solutions/7480222/kotlin-rust-by-samoylenkodmitry-2qdw/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/09012026-865-smallest-subtree-with?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/tWByRVZUQd4)

![70318a61-7859-48b4-9d91-94eef697bd89 (1).webp](/assets/leetcode_daily_images/6ebbc250.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1232

#### Problem TLDR

Lowest common ancestor of the deepest nodes #medium

#### Intuition

Do a DFS.
One way: propagate depth down, compare on the way up.
Second way: return both depth & result on the way up.

#### Approach

* how to use uniqness?

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(logn)$$

#### Code

```kotlin
// 1ms
    fun subtreeWithAllDeepest(r: TreeNode?): TreeNode? {
        fun dfs(n: TreeNode?): Pair<Int, TreeNode?> = n?.run {
            val (l, a) = dfs(left); val (r, b) = dfs(right)
            (1 + max(l, r)) to if (l > r) a else if (l < r) b else n
        } ?: 0 to null
        return dfs(r).second
    }
```
```rust
// 0ms
    pub fn subtree_with_all_deepest(r: Option<Rc<RefCell<TreeNode>>>) -> Option<Rc<RefCell<TreeNode>>> {
        fn dfs(ro: &Option<Rc<RefCell<TreeNode>>>, d: i32, max: &mut i32, res: &mut Option<Rc<RefCell<TreeNode>>>) -> i32 {
            let Some(n) = ro else { return d }; let n = n.borrow();
            let (l,r) = (dfs(&n.left, d+1, max, res), dfs(&n.right, d+1, max, res));  *max = l.max(r).max(*max);
            if l == *max && r == *max { *res = ro.clone() }; l.max(r)
        }
        let (mut max, mut res) = (0, None); dfs(&r, 0, &mut max, &mut res); res
    }
```

