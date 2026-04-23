---
layout: leetcode-entry
title: "1110. Delete Nodes And Return Forest"
permalink: "/leetcode/problem/2024-07-17-1110-delete-nodes-and-return-forest/"
leetcode_ui: true
entry_slug: "2024-07-17-1110-delete-nodes-and-return-forest"
---

[1110. Delete Nodes And Return Forest](https://leetcode.com/problems/delete-nodes-and-return-forest/description/) medium
[blog post](https://leetcode.com/problems/delete-nodes-and-return-forest/solutions/5489110/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/17072024-1110-delete-nodes-and-return?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/C9CDLWLa3mo)
![2024-07-17_08-51.webp](/assets/leetcode_daily_images/99f49ec3.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/673

#### Problem TLDR

Trees after remove nodes from tree #medium #tree

#### Intuition

Just iterate and remove on the fly in a single Depth-First Search. Use a HashSet for O(1) checks.

#### Approach

* code looks nicer when we can do `n.left = dfs(n.left)`
* Rust's `Option` clone() is cheap

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun delNodes(root: TreeNode?, to_delete: IntArray) = buildList {
        val set = to_delete.toSet()
        fun dfs(n: TreeNode?): TreeNode? = n?.run {
            left = dfs(left); right = dfs(right); val remove = `val` in set
            if (remove) { left?.let(::add); right?.let(::add) }
            takeIf { !remove }
        }
        dfs(root)?.let(::add)
    }

```
```rust

    pub fn del_nodes(root: Option<Rc<RefCell<TreeNode>>>, to_delete: Vec<i32>) -> Vec<Option<Rc<RefCell<TreeNode>>>> {
        let set: HashSet<_> = to_delete.into_iter().collect(); let mut res = vec![];
        fn dfs(n_opt: &Option<Rc<RefCell<TreeNode>>>, set: &HashSet<i32>, res: &mut Vec<Option<Rc<RefCell<TreeNode>>>>)
            -> Option<Rc<RefCell<TreeNode>>> {
                let Some(n_rc) = n_opt else { return None }; let mut n = n_rc.borrow_mut();
                n.left = dfs(&n.left, set, res); n.right = dfs(&n.right, set, res);
                if set.contains(&n.val) {
                    if n.left.is_some() { res.push(n.left.clone()); }; if n.right.is_some() { res.push(n.right.clone()); }
                    None
                } else { (*n_opt).clone() }
            }
        let root = dfs(&root, &set, &mut res); if root.is_some() { res.push(root) }; res
    }

```

